
Author : Senior Python Developer / Sports Data Scientist
Version : 2.0.0
License : MIT
Description

A production-ready, statistically robust football (soccer) match prediction
engine that combines:
Poisson goal-distribution modelling
Monte Carlo simulation (10,000 iterations per match)
Machine learning ensemble (Random Forest + Gradient Boosting + Logistic
Regression) with soft-voting meta-classifier
Dixon-Coles attack/defence strength estimation
Form weighting, home-advantage factor, and xG integration
Data Sources
------------
The script ships with a rich synthetic-but-realistic dataset that mirrors
real Premier League / Champions League statistics (form, xG, shots, etc.).
To plug in live data, replace the `DataProvider` class methods with API calls
to football-data.org, API-Football, or Opta / StatsBomb feeds.
Usage
-----
python football_predictor.py
Output
------
Console report + JSON result saved to football_predictions.json
Dependencies
------------
pandas, numpy, scikit-learn, scipy (all standard; no network needed)
================================================================================
"""
from __future__ import annotations
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.ensemble import (
GradientBoostingClassifier,
RandomForestClassifier,
VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
#
#
# Logging
logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(levelname)-8s %(message)s",
datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
#
# Data Structures
#
@dataclass
class MatchResult:
"""Container for one historical match result."""
home_team: str
away_team: str
home_goals: int
away_goals: int
home_xg: float = 0.0
away_xg: float = 0.0
home_shots: int = 0
away_shots: int = 0
home_shots_on_target: int = 0
away_shots_on_target: int = 0
home_possession: float = 50.0
away_possession: float = 50.0
competition: str = "Premier League"
matchday: int = 1
@dataclass
class TeamProfile:
"""Aggregated team statistics derived from recent fixtures."""
name: str
attack_strength: float = 1.0
defence_strength: float = 1.0
home_attack: float = 1.0
home_defence: float = 1.0
away_attack: float = 1.0
away_defence: float = 1.0
avg_goals_scored: float = 1.5
avg_goals_conceded: float = 1.2
avg_xg_for: float = 1.5
avg_xg_against: float = 1.2
avg_shots: float = 12.0
avg_shots_conceded: float = 10.0
avg_possession: float = 50.0
form_score: float = 0.5 # 0 = worst, 1 = best
last_10: List[str] = field(default_factory=list) # 'W','D','L'
@dataclass
class PredictionReport:
"""Full prediction output for a single fixture."""
home_team: str
away_team: str
# Outcome probabilities
home_win_prob: float = 0.0
draw_prob: float = 0.0
away_win_prob: float = 0.0
# Goals market
over_25_prob: float = 0.0
under_25_prob: float = 0.0
over_35_prob: float = 0.0
btts_prob: float = 0.0
# Score predictions
expected_home_goals: float = 0.0
expected_away_goals: float = 0.0
most_likely_scores: List[Tuple[int, int, float]] = field(default_factory=list)
# ML ensemble
ml_home_win_prob: float = 0.0
ml_draw_prob: float = 0.0
ml_away_win_prob: float = 0.0
# Combined final
predicted_winner: str = ""
confidence: float = 0.0
# Raw sims
simulation_count: int = 10_000
#
#
# Synthetic Data Provider
class DataProvider:
"""
Supplies historical match data and team statistics.
Replace `_build_synthetic_dataset()` with live API calls
(football-data.org, API-Football, StatsBomb, etc.) to operate
on real fixtures.
"""
# League-wide averages used when estimating attack/defence strength
LEAGUE_AVG_HOME_GOALS: float = 1.55
LEAGUE_AVG_AWAY_GOALS: float = 1.20
def __init__(self) -> None:
self.matches: List[MatchResult] = self._build_synthetic_dataset()
self.df: pd.DataFrame = self._to_dataframe()
log.info("DataProvider: loaded %d historical fixtures.", len(self.matches))
# ------------------------------------------------------------------ #
# Public API #
# ------------------------------------------------------------------ #
def get_team_profile(self, team: str, n_recent: int = 10) -> TeamProfile:
"""Return a `TeamProfile` for *team* using the last *n_recent* matches."""
home_m = self.df[self.df.home_team == team].tail(n_recent)
away_m = self.df[self.df.away_team == team].tail(n_recent)
# Combine home & away appearances
all_m = pd.concat([
home_m.assign(
goals_for=home_m.home_goals,
goals_against=home_m.away_goals,
xg_for=home_m.home_xg,
xg_against=home_m.away_xg,
shots_for=home_m.home_shots,
shots_against=home_m.away_shots,
possession=home_m.home_possession,
result=np.where(home_m.home_goals > home_m.away_goals, "W",
np.where(home_m.home_goals == home_m.away_goals, "D", "L"))
),
away_m.assign(
goals_for=away_m.away_goals,
goals_against=away_m.home_goals,
xg_for=away_m.away_xg,
xg_against=away_m.home_xg,
shots_for=away_m.away_shots,
shots_against=away_m.home_shots,
possession=away_m.away_possession,
result=np.where(away_m.away_goals > away_m.home_goals, "W",
np.where(away_m.away_goals == away_m.home_goals, "D", "L"))
),
]).sort_values("matchday").tail(n_recent)
if all_m.empty:
log.warning("No data found for '%s'; using league averages.", team)
return TeamProfile(name=team)
avg_gf = all_m.goals_for.mean()
avg_ga = all_m.goals_against.mean()
avg_xgf = all_m.xg_for.mean()
avg_xga = all_m.xg_against.mean()
avg_shot = all_m.shots_for.mean()
avg_sc = all_m.shots_against.mean()
avg_poss = all_m.possession.mean()
last_10 = all_m.result.tolist()[-10:]
# Form score: weighted W=1, D=0.5, L=0 (recent matches weighted higher)
weights = np.linspace(0.5, 1.0, len(last_10))
form_values = np.array([1.0 if r == "W" else 0.5 if r == "D" else 0.0
for r in last_10])
form_score = float(np.dot(form_values, weights) / weights.sum())
# Dixon-Coles style attack/defence strength (relative to league avg)
att_str = avg_gf / self.LEAGUE_AVG_HOME_GOALS
def_str = avg_ga / self.LEAGUE_AVG_AWAY_GOALS
# Home / Away split
h_gf = home_m.home_goals.mean() if not home_m.empty else avg_gf
h_ga = home_m.away_goals.mean() if not home_m.empty else avg_ga
a_gf = away_m.away_goals.mean() if not away_m.empty else avg_gf
a_ga = away_m.home_goals.mean() if not away_m.empty else avg_ga
h_att = h_gf / self.LEAGUE_AVG_HOME_GOALS
h_def = h_ga / self.LEAGUE_AVG_AWAY_GOALS
a_att = a_gf / self.LEAGUE_AVG_AWAY_GOALS
a_def = a_ga / self.LEAGUE_AVG_HOME_GOALS
return TeamProfile(
name=team,
attack_strength=att_str,
defence_strength=def_str,
home_attack=h_att,
home_defence=h_def,
away_attack=a_att,
away_defence=a_def,
avg_goals_scored=avg_gf,
avg_goals_conceded=avg_ga,
avg_xg_for=avg_xgf,
avg_xg_against=avg_xga,
avg_shots=avg_shot,
avg_shots_conceded=avg_sc,
avg_possession=avg_poss,
form_score=form_score,
last_10=last_10,
)
def get_historical_dataframe(self) -> pd.DataFrame:
return self.df.copy()
# ------------------------------------------------------------------ #
# Internal helpers #
# ------------------------------------------------------------------ #
def _to_dataframe(self) -> pd.DataFrame:
rows = [asdict(m) for m in self.matches]
return pd.DataFrame(rows)
def _build_synthetic_dataset(self) -> List[MatchResult]:
"""
Generate 300 realistic synthetic fixtures across 20 Premier League-style
clubs. Statistical parameters are calibrated to real PL distributions:
- Home avg goals 1.55, Away avg goals 1.20
- xG tracks actual goals with 0.3 noise
- Shots: home 1215, away 912
- Possession: home 5060 %
"""
rng = np.random.default_rng(seed=42)
teams = {
# team_name: (attack_mu, defence_mu) "Manchester City": (2.30, 0.85),
controls goal-scoring tendency
"Arsenal": (2.10, 0.95),
"Liverpool": (2.20, 1.00),
"Chelsea": (1.80, 1.10),
"Tottenham": (1.85, 1.30),
"Manchester United": (1.60, 1.35),
"Newcastle United": (1.70, 1.15),
"Aston Villa": (1.65, 1.20),
"West Ham": (1.50, 1.40),
"Brighton": (1.55, 1.20),
"Fulham": (1.35, 1.30),
"Brentford": (1.45, 1.35),
"Crystal Palace": (1.25, 1.25),
"Wolverhampton": (1.20, 1.30),
"Everton": (1.15, 1.40),
"Nottm Forest": (1.20, 1.15),
"Bournemouth": (1.30, 1.45),
"Burnley": (0.95, 1.60),
"Sheffield United": (0.90, 1.65),
"Luton Town": (1.00, 1.55),
}
team_names = list(teams.keys())
records: List[MatchResult] = []
matchday = 1
# Round-robin (each pair plays twice: home & away)
for i, home in enumerate(team_names):
for j, away in enumerate(team_names):
if i == j:
continue
h_att, h_def = teams[home]
a_att, a_def = teams[away]
# Poisson lambda with home advantage (+18 %)
lam_h = max(0.3, h_att * (1 / a_def) * 1.18)
lam_a = max(0.2, a_att * (1 / h_def))
hg = int(rng.poisson(lam_h))
ag = int(rng.poisson(lam_a))
# xG = actual goals noise, clipped to [0.2, 5.0]
h_xg = float(np.clip(hg + rng.normal(0, 0.25), 0.2, 5.0))
a_xg = float(np.clip(ag + rng.normal(0, 0.25), 0.2, 5.0))
h_shots = int(rng.integers(8, 18))
a_shots = int(rng.integers(6, 15))
h_sot = int(np.clip(rng.integers(2, h_shots), 2, h_shots))
a_sot = int(np.clip(rng.integers(1, a_shots), 1, a_shots))
h_poss = float(np.clip(rng.normal(52, 7), 35, 70))
a_poss = round(100.0 - h_poss, 1)
records.append(MatchResult(
home_team=home,
away_team=away,
home_goals=hg,
away_goals=ag,
home_xg=round(h_xg, 2),
away_xg=round(a_xg, 2),
home_shots=h_shots,
away_shots=a_shots,
home_shots_on_target=h_sot,
away_shots_on_target=a_sot,
home_possession=round(h_poss, 1),
away_possession=a_poss,
competition="Premier League",
matchday=matchday,
))
matchday += 1
return records
#
#
# Poisson / Dixon-Coles Goal Model
class PoissonGoalModel:
"""
Estimates expected goals for each team using a bivariate Poisson model
with a Dixon-Coles low-score correction (rho parameter).
_home = _home _away _home_advantage
_away = _away _home
where = attack, = defence (lower is better), = home advantage.
"""
HOME_ADVANTAGE: float = 1.20 # multiplicative boost for home side
def __init__(
self,
home_profile: TeamProfile,
away_profile: TeamProfile,
league_avg_home: float = 1.55,
league_avg_away: float = 1.20,
) -> None:
self.home = home_profile
self.away = away_profile
self.league_h = league_avg_home
self.league_a = league_avg_away
self.lambda_home, self.lambda_away = self._compute_lambdas()
def _compute_lambdas(self) -> Tuple[float, float]:
"""
Blend raw form-based goal averages with Dixon-Coles attack/defence
strength ratios, then apply home-advantage multiplier.
"""
# --- Strength-based estimate ---
str_home = (
self.home.home_attack
* self.away.away_defence
* self.league_h
* self.HOME_ADVANTAGE
)
str_away = (
self.away.away_attack
* self.home.home_defence
* self.league_a
)
# --- Form-based estimate (last-10 goal average) ---
form_home = self.home.avg_goals_scored * self.HOME_ADVANTAGE
form_away = self.away.avg_goals_scored
# --- xG blend (if non-trivial) ---
xg_home = (self.home.avg_xg_for + (1 - self.away.avg_xg_against / self.league_h)) * s
xg_away = (self.away.avg_xg_for + (1 - self.home.avg_xg_against / self.league_a))
# Weighted blend: strength 40 %, form 35 %, xG 25 %
lam_h = 0.40 * str_home + 0.35 * form_home + 0.25 * xg_home
lam_a = 0.40 * str_away + 0.35 * form_away + 0.25 * xg_away
# Form-score adjustment (10 %)
lam_h *= 1.0 + 0.10 * (self.home.form_score - 0.5)
lam_a *= 1.0 + 0.10 * (self.away.form_score - 0.5)
return max(0.20, round(lam_h, 4)), max(0.10, round(lam_a, 4))
@staticmethod
def _dixon_coles_correction(hg: int, ag: int, lam_h: float, lam_a: float,
rho: float = -0.10) -> float:
"""
Applies the Dixon-Coles (1997) low-score correction factor ,
which adjusts the joint probability for scores (0-0), (1-0), (0-1), (1-1).
"""
if hg == 0 and ag == 0:
return 1.0 - lam_h * lam_a * rho
elif hg == 1 and ag == 0:
return 1.0 + lam_a * rho
elif hg == 0 and ag == 1:
return 1.0 + lam_h * rho
elif hg == 1 and ag == 1:
return 1.0 - rho
return 1.0
def score_probability(self, home_goals: int, away_goals: int,
rho: float = -0.10) -> float:
"""P(home scores h, away scores a) under Poisson + DC correction."""
p = (poisson.pmf(home_goals, self.lambda_home) *
poisson.pmf(away_goals, self.lambda_away) *
self._dixon_coles_correction(home_goals, away_goals,
self.lambda_home, self.lambda_away, rho))
return max(0.0, p)
def outcome_probabilities(self, max_goals: int = 10) -> Tuple[float, float, float]:
"""Return (P_home_win, P_draw, P_away_win) by summing score grid."""
p_home = p_draw = p_away = 0.0
for h in range(max_goals + 1):
for a in range(max_goals + 1):
p = self.score_probability(h, a)
if h > a:
p_home += p
elif h == a:
p_draw += p
else:
p_away += p
total = p_home + p_draw + p_away
if total == 0:
return 0.33, 0.33, 0.34
return p_home / total, p_draw / total, p_away / total
def top_scorelines(self, n: int = 8, max_goals: int = 8
) -> List[Tuple[int, int, float]]:
"""Return the top-n most probable correct scores."""
scores = []
for h in range(max_goals + 1):
for a in range(max_goals + 1):
scores.append((h, a, self.score_probability(h, a)))
scores.sort(key=lambda x: x[2], reverse=True)
return scores[:n]
#
#
# Monte Carlo Simulation Engine
class MonteCarloSimulator:
"""
Runs N independent match simulations by drawing home/away goals from
independent Poisson distributions calibrated to _home and _away.
Markets estimated:
1X2 (home win / draw / away win)
Over/Under 2.5 and 3.5 goals
Both Teams to Score (BTTS)
Correct score distribution
"""
def __init__(self, lambda_home: float, lambda_away: float,
n_simulations: int = 10_000, seed: int = 0) -> None:
self.lam_h = lambda_home
self.lam_a = lambda_away
self.n = n_simulations
self.rng = np.random.default_rng(seed=seed)
self._results: Optional[pd.DataFrame] = None
def run(self) -> "MonteCarloSimulator":
"""Execute all simulations and cache results."""
home_goals = self.rng.poisson(self.lam_h, self.n)
away_goals = self.rng.poisson(self.lam_a, self.n)
total_goals = home_goals + away_goals
outcomes = np.where(home_goals > away_goals, "H",
np.where(home_goals == away_goals, "D", "A"))
self._results = pd.DataFrame({
"home_goals": home_goals,
"away_goals": away_goals,
"total_goals": total_goals,
"outcome": outcomes,
})
log.debug("Monte Carlo: %d simulations completed (H=%.3f, A=%.3f).",
self.n, self.lam_h, self.lam_a)
return self
# ------------------------------------------------------------------ #
# Market probabilities #
# ------------------------------------------------------------------ #
@property
def outcome_probs(self) -> Dict[str, float]:
df = self._require_results()
vc = df["outcome"].value_counts(normalize=True)
return {
"home_win": float(vc.get("H", 0.0)),
"draw": float(vc.get("D", 0.0)),
"away_win": float(vc.get("A", 0.0)),
}
@property
def over_25_prob(self) -> float:
df = self._require_results()
return float((df["total_goals"] > 2.5).mean())
@property
def under_25_prob(self) -> float:
return 1.0 - self.over_25_prob
@property
def over_35_prob(self) -> float:
df = self._require_results()
return float((df["total_goals"] > 3.5).mean())
@property
def btts_prob(self) -> float:
df = self._require_results()
return float(((df["home_goals"] > 0) & (df["away_goals"] > 0)).mean())
@property
def top_scores(self) -> List[Tuple[int, int, float]]:
df = self._require_results()
score_counts = df.groupby(["home_goals", "away_goals"]).size().reset_index(name="coun
score_counts["prob"] = score_counts["count"] / self.n
score_counts.sort_values("prob", ascending=False, inplace=True)
return [
(int(r.home_goals), int(r.away_goals), round(float(r.prob), 4))
for _, r in score_counts.head(8).iterrows()
]
# ------------------------------------------------------------------ #
# Internal #
# ------------------------------------------------------------------ #
def _require_results(self) -> pd.DataFrame:
if self._results is None:
raise RuntimeError("Call .run() before accessing results.")
return self._results
#
#
# Feature Engineering
class FeatureEngineer:
"""
Transforms raw historical match data into a feature matrix suitable
for supervised ML models.
Features per fixture row:
home_attack_str, home_defence_str, away_attack_str, away_defence_str,
home_form, away_form, home_avg_goals, away_avg_goals,
home_avg_conceded, away_avg_conceded, home_xg, away_xg,
home_shots, away_shots, home_possession,
goal_diff_proxy, xg_diff, shots_diff
Target:
outcome 0=home win, 1=draw, 2=away win
"""
FEATURE_COLS = [
"home_attack_str", "home_defence_str",
"away_attack_str", "away_defence_str",
"home_form", "away_form",
"home_avg_goals", "away_avg_goals",
"home_avg_conceded", "away_avg_conceded",
"home_xg", "away_xg",
"home_shots", "away_shots",
"home_possession",
"goal_diff_proxy", "xg_diff", "shots_diff",
]
def __init__(self, data_provider: DataProvider) -> None:
self.provider = data_provider
def build_feature_matrix(self) -> Tuple[pd.DataFrame, pd.Series]:
"""Build X (features) and y (outcome labels) from historical data."""
df = self.provider.get_historical_dataframe()
rows = []
for _, row in df.iterrows():
hp = self.provider.get_team_profile(row.home_team)
ap = self.provider.get_team_profile(row.away_team)
rows.append({
"home_attack_str": hp.attack_strength,
"home_defence_str": hp.defence_strength,
"away_attack_str": ap.attack_strength,
"away_defence_str": ap.defence_strength,
"home_form": hp.form_score,
"away_form": ap.form_score,
"home_avg_goals": hp.avg_goals_scored,
"away_avg_goals": ap.avg_goals_scored,
"home_avg_conceded": hp.avg_goals_conceded,
"away_avg_conceded": ap.avg_goals_conceded,
"home_xg": hp.avg_xg_for,
"away_xg": ap.avg_xg_for,
"home_shots": hp.avg_shots,
"away_shots": ap.avg_shots,
"home_possession": hp.avg_possession,
"goal_diff_proxy": hp.avg_goals_scored - ap.avg_goals_scored,
"xg_diff": hp.avg_xg_for - ap.avg_xg_for,
"shots_diff": hp.avg_shots - ap.avg_shots,
# target
"_hg": row.home_goals,
"_ag": row.away_goals,
})
result_df = pd.DataFrame(rows)
# Encode outcome
result_df["outcome"] = np.where(
result_df["_hg"] > result_df["_ag"], 0,
np.where(result_df["_hg"] == result_df["_ag"], 1, 2),
)
X = result_df[self.FEATURE_COLS].fillna(0.0)
y = result_df["outcome"]
return X, y
def features_for_fixture(self, home: TeamProfile, away: TeamProfile,
poisson_model: PoissonGoalModel) -> pd.DataFrame:
"""Build a single-row feature DataFrame for an upcoming fixture."""
row = {
"home_attack_str": home.attack_strength,
"home_defence_str": home.defence_strength,
"away_attack_str": away.attack_strength,
"away_defence_str": away.defence_strength,
"home_form": home.form_score,
"away_form": away.form_score,
"home_avg_goals": home.avg_goals_scored,
"away_avg_goals": away.avg_goals_scored,
"home_avg_conceded": home.avg_goals_conceded,
"away_avg_conceded": away.avg_goals_conceded,
"home_xg": home.avg_xg_for,
"away_xg": away.avg_xg_for,
"home_shots": home.avg_shots,
"away_shots": away.avg_shots,
"home_possession": home.avg_possession,
"goal_diff_proxy": home.avg_goals_scored - away.avg_goals_scored,
"xg_diff": home.avg_xg_for - away.avg_xg_for,
"shots_diff": home.avg_shots - away.avg_shots,
}
return pd.DataFrame([row])[self.FEATURE_COLS]
#
#
# ML Ensemble Classifier
class MLEnsemble:
"""
Soft-voting ensemble of:
Random Forest (100 trees, max_depth 6)
Gradient Boosting (200 estimators, lr 0.05)
Logistic Regression (L2, C=1.0, multi-class='multinomial')
The scaler is fit once on training data and applied to all subsequent
prediction calls.
"""
def __init__(self) -> None:
rf = RandomForestClassifier(
n_estimators=150, max_depth=6, min_samples_leaf=4,
random_state=42, n_jobs=-1
)
gb = GradientBoostingClassifier(
n_estimators=200, learning_rate=0.05, max_depth=4,
subsample=0.8, random_state=42
)
lr = LogisticRegression(
C=1.0, solver="lbfgs",
max_iter=1000, random_state=42
)
self.model = VotingClassifier(
estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
voting="soft",
weights=[2, 2, 1], # slightly favour tree-based models
)
self.scaler = StandardScaler()
self._trained = False
self.cv_accuracy: float = 0.0
def train(self, X: pd.DataFrame, y: pd.Series) -> "MLEnsemble":
X_arr = self.scaler.fit_transform(X.values)
X_tr, X_te, y_tr, y_te = train_test_split(
X_arr, y, test_size=0.20, random_state=42, stratify=y
)
self.model.fit(X_tr, y_tr)
y_pred = self.model.predict(X_te)
hold_acc = accuracy_score(y_te, y_pred)
cv_scores = cross_val_score(self.model, X_arr, y, cv=5, scoring="accuracy")
self.cv_accuracy = float(cv_scores.mean())
self._trained = True
log.info(
"ML Ensemble trained | Hold-out acc: %.2f%% | 5-fold CV: %.2f%% %.2f%%",
hold_acc * 100, self.cv_accuracy * 100, cv_scores.std() * 100
)
return self
def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
"""Return class probabilities [P(home), P(draw), P(away)]."""
if not self._trained:
raise RuntimeError("Model must be trained before prediction.")
X_sc = self.scaler.transform(X.values)
return self.model.predict_proba(X_sc)[0] # single row
#
#
# Prediction Engine (orchestrator)
class PredictionEngine:
"""
Orchestrates the full prediction pipeline for a list of fixtures.
Pipeline:
1. Load & aggregate historical data (DataProvider)
2. Train ML ensemble (MLEnsemble)
3. For each fixture:
a. Build team profiles (DataProvider)
b. Compute Poisson values (PoissonGoalModel)
c. Run Monte Carlo simulation (10 000 draws) (MonteCarloSimulator)
d. Extract ML probabilities (MLEnsemble)
e. Blend Poisson + MC + ML (weighted average)
f. Assemble PredictionReport
"""
# Blend weights for final outcome probabilities:
# Poisson 35 % + Monte Carlo 35 % + ML 30 %
W_POISSON = 0.35
W_MC = 0.35
W_ML = 0.30
def __init__(self, n_simulations: int = 10_000) -> None:
self.n_simulations = n_simulations
log.info("Initialising PredictionEngine (Monte Carlo N=%d)", n_simulations)
self.provider = DataProvider()
self.feat_eng = FeatureEngineer(self.provider)
self.ml_model = MLEnsemble()
X, y = self.feat_eng.build_feature_matrix()
self.ml_model.train(X, y)
# ------------------------------------------------------------------ #
# Public entry-point #
# ------------------------------------------------------------------ #
def predict(self, fixtures: List[Tuple[str, str]]) -> List[PredictionReport]:
"""
Predict outcomes for a list of (home_team, away_team) tuples.
Returns a list of `PredictionReport` objects in the same order.
"""
reports = []
for home_name, away_name in fixtures:
log.info("Predicting: %s vs %s", home_name, away_name)
try:
report = self._predict_fixture(home_name, away_name)
reports.append(report)
except Exception as exc:
log.error("Failed to predict %s vs %s: %s", home_name, away_name, exc)
return reports
# ------------------------------------------------------------------ #
# Core prediction for one fixture #
# ------------------------------------------------------------------ #
def _predict_fixture(self, home_name: str, away_name: str) -> PredictionReport:
# 1 team profiles
hp = self.provider.get_team_profile(home_name)
ap = self.provider.get_team_profile(away_name)
# 2 Poisson model
poisson_model = PoissonGoalModel(hp, ap)
p_poisson = poisson_model.outcome_probabilities() # (H, D, A)
# 3 Monte Carlo
mc = MonteCarloSimulator(
lambda_home=poisson_model.lambda_home,
lambda_away=poisson_model.lambda_away,
n_simulations=self.n_simulations,
).run()
mc_probs = mc.outcome_probs
# 4 ML ensemble
feat_row = self.feat_eng.features_for_fixture(hp, ap, poisson_model)
ml_proba = self.ml_model.predict_proba(feat_row) # [H, D, A]
# 5 Blend
blended_h = (self.W_POISSON * p_poisson[0] +
self.W_MC * mc_probs["home_win"] +
self.W_ML * ml_proba[0])
blended_d = (self.W_POISSON * p_poisson[1] +
self.W_MC * mc_probs["draw"] +
self.W_ML * ml_proba[1])
blended_a = (self.W_POISSON * p_poisson[2] +
self.W_MC * mc_probs["away_win"] +
self.W_ML * ml_proba[2])
# Normalise to sum = 1
total = blended_h + blended_d + blended_a
blended_h /= total
blended_d /= total
blended_a /= total
# 6 Determine predicted winner
probs = {"home": blended_h, "draw": blended_d, "away": blended_a}
predicted = max(probs, key=probs.__getitem__)
if predicted == "home":
winner_label = home_name
elif predicted == "away":
winner_label = away_name
else:
winner_label = "Draw"
# Top scores: blend Poisson grid and Monte Carlo distribution
poisson_scores = {(h, a): p for h, a, p in poisson_model.top_scorelines(n=20)}
mc_scores = {(h, a): p for h, a, p in mc.top_scores}
all_keys = set(poisson_scores) | set(mc_scores)
merged = {}
for k in all_keys:
merged[k] = 0.5 * poisson_scores.get(k, 0.0) + 0.5 * mc_scores.get(k, 0.0)
top_scores = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:8]
top_scores_list = [(h, a, round(p, 4)) for (h, a), p in top_scores]
return PredictionReport(
home_team=home_name,
away_team=away_name,
home_win_prob=round(blended_h, 4),
draw_prob=round(blended_d, 4),
away_win_prob=round(blended_a, 4),
over_25_prob=round(mc.over_25_prob, 4),
under_25_prob=round(mc.under_25_prob, 4),
over_35_prob=round(mc.over_35_prob, 4),
btts_prob=round(mc.btts_prob, 4),
expected_home_goals=round(poisson_model.lambda_home, 3),
expected_away_goals=round(poisson_model.lambda_away, 3),
most_likely_scores=top_scores_list,
ml_home_win_prob=round(float(ml_proba[0]), 4),
ml_draw_prob=round(float(ml_proba[1]), 4),
ml_away_win_prob=round(float(ml_proba[2]), 4),
predicted_winner=winner_label,
confidence=round(max(blended_h, blended_d, blended_a), 4),
simulation_count=self.n_simulations,
)
#
#
# Report Formatter
class ReportFormatter:
"""Renders prediction reports to the console in a readable format."""
SEP = "" * 72
@staticmethod
def print_report(report: PredictionReport) -> None:
r = report
print()
print(ReportFormatter.SEP)
print(f" {r.home_team:25s} vs {r.away_team}")
print(ReportFormatter.SEP)
print(f"\n OUTCOME PROBABILITIES (Blended: Poisson + MC + ML)")
print(f" {'Home Win':<20} {r.home_win_prob*100:6.2f}%")
print(f" {'Draw':<20} {r.draw_prob*100:6.2f}%")
print(f" {'Away Win':<20} {r.away_win_prob*100:6.2f}%")
print(f"\n ML ENSEMBLE PROBABILITIES")
print(f" {'Home Win':<20} {r.ml_home_win_prob*100:6.2f}%")
print(f" {'Draw':<20} {r.ml_draw_prob*100:6.2f}%")
print(f" {'Away Win':<20} {r.ml_away_win_prob*100:6.2f}%")
print(f"\n GOALS MARKETS (based on {r.simulation_count:,} simulations)")
print(f" {'xG Home':<20} {r.expected_home_goals:.3f}")
print(f" {'xG Away':<20} {r.expected_away_goals:.3f}")
print(f" {'Over 2.5':<20} {r.over_25_prob*100:6.2f}%")
print(f" {'Under 2.5':<20} {r.under_25_prob*100:6.2f}%")
print(f" {'Over 3.5':<20} {r.over_35_prob*100:6.2f}%")
print(f" {'BTTS':<20} {r.btts_prob*100:6.2f}%")
print(f"\n TOP CORRECT SCORE PROBABILITIES")
for h, a, p in r.most_likely_scores[:6]:
bar = "" * int(p * 200)
print(f" {h} {a} {p*100:5.2f}% {bar}")
print(f"\n PREDICTION: {r.predicted_winner.upper()} "
f"(confidence {r.confidence*100:.1f}%)")
print()
@staticmethod
def save_json(reports: List[PredictionReport], path: str = "football_predictions.json") -
data = [asdict(r) for r in reports]
with open(path, "w", encoding="utf-8") as fh:
json.dump(data, fh, indent=2, ensure_ascii=False)
log.info("Predictions saved %s", path)
#
# Main
#
FIXTURES: List[Tuple[str, str]] = [
# Format: (Home Team, Away Team)
("Manchester City", "Arsenal"),
("Liverpool", "Chelsea"),
("Tottenham", "Manchester United"),
("Newcastle United", "Aston Villa"),
("Brighton", "West Ham"),
("Brentford", "Crystal Palace"),
]
def main() -> None:
print()
print("=" * 72)
print(" ADVANCED FOOTBALL PREDICTION SYSTEM v2.0")
print(" Poisson + Monte Carlo + ML Ensemble")
print("=" * 72)
# Build engine (trains ML model internally)
engine = PredictionEngine(n_simulations=10_000)
# Predict all fixtures
reports = engine.predict(FIXTURES)
# Print to console
for report in reports:
ReportFormatter.print_report(report)
# Persist to JSON
ReportFormatter.save_json(reports, "/mnt/user-data/outputs/football_predictions.json")
print("=" * 72)
print(f" Analysis complete {len(reports)} fixtures predicted.")
print("=" * 72)
print()
if __name__ == "__main__":

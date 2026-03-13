import os
from flask import Flask, request, jsonify
from football_predictor import PredictionEngine

app = Flask(__name__)

print("Loading prediction engine...")
engine = PredictionEngine(n_simulations=10_000)
print("Engine ready!")

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Football Prediction API is running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "matches" not in data:
        return jsonify({"error": "Please provide matches"}), 400
    fixtures = [(m["home"], m["away"]) for m in data["matches"]]
    reports = engine.predict(fixtures)
    results = []
    for r in reports:
        results.append({
            "home_team": r.home_team,
            "away_team": r.away_team,
            "home_win_prob": round(r.home_win_prob * 100, 2),
            "draw_prob": round(r.draw_prob * 100, 2),
            "away_win_prob": round(r.away_win_prob * 100, 2),
            "over_25_prob": round(r.over_25_prob * 100, 2),
            "under_25_prob": round(r.under_25_prob * 100, 2),
            "over_35_prob": round(r.over_35_prob * 100, 2),
            "btts_prob": round(r.btts_prob * 100, 2),
            "expected_home_xg": r.expected_home_goals,
            "expected_away_xg": r.expected_away_goals,
            "predicted_winner": r.predicted_winner,
            "confidence": round(r.confidence * 100, 2),
            "top_scores": [
                {"home": h, "away": a, "probability": round(p * 100, 2)}
                for h, a, p in r.most_likely_scores
            ],
        })
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

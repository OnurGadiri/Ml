import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request


def main():
    a = Flask(__name__)

    b = Path("models")
    c = joblib.load(b / "scaler.pkl")
    d = joblib.load(b / "classifier.pkl")

    with open(b / "metrics.json", encoding="utf-8") as e:
        f = json.load(e)
    g = f["classes"]

    @a.route("/health", methods=["GET"])
    def h():
        return jsonify({"status": "ok", "model": f["best_model"]})

    @a.route("/predict", methods=["POST"])
    def i():
        j = request.get_json(silent=True)
        if not j:
            return jsonify({"error": "send json body"}), 400

        k = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        l = []
        for m in k:
            if m not in j:
                return jsonify({"error": f"missing field: {m}"}), 400
            l.append(float(j[m]))

        n = np.array(l).reshape(1, -1)
        o = c.transform(n)
        p = d.predict(o)[0]
        q = d.predict_proba(o)[0]
        r = g[p]

        return jsonify(
            {
                "species": r,
                "confidence": round(float(q[p]), 4),
                "probabilities": {g[s]: round(float(q[s]), 4) for s in range(len(g))},
            }
        )

    print("server: http://127.0.0.1:5000")
    a.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory


def main():
    a = Flask(__name__, static_folder="static", static_url_path="/static")

    b = Path("models")
    c = joblib.load(b / "scaler.pkl")
    d = joblib.load(b / "classifier.pkl")

    with open(b / "metrics.json", encoding="utf-8") as e:
        f = json.load(e)
    g = f["classes"]
    h = f.get("feature_keys", ["sepal_length", "sepal_width", "petal_length", "petal_width"])
    i = Path("data/predictions.json")
    j = Path("data/stats.json")

    def k():
        if not i.exists():
            return []
        with open(i, encoding="utf-8") as l:
            return json.load(l)

    def m(n):
        i.parent.mkdir(exist_ok=True)
        o = k()
        o.insert(0, n)
        o = o[:50]
        with open(i, "w", encoding="utf-8") as p:
            json.dump(o, p, indent=2)

    @a.route("/")
    def t():
        return send_from_directory("static", "index.html")

    @a.route("/chart/<q>")
    def u(q):
        r = Path("output") / f"{q}.png"
        if not r.exists():
            return jsonify({"error": "chart not found"}), 404
        return send_from_directory("output", f"{q}.png")

    @a.route("/health", methods=["GET"])
    def v():
        return jsonify({"status": "ok", "model": f["best_model"]})

    @a.route("/dataset", methods=["GET"])
    def w():
        if not j.exists():
            return jsonify({"error": "run expand_data.py first"}), 404
        with open(j, encoding="utf-8") as x:
            return jsonify(json.load(x))

    @a.route("/metrics", methods=["GET"])
    def y():
        return jsonify(f)

    @a.route("/history", methods=["GET"])
    def z():
        return jsonify(k())

    @a.route("/samples", methods=["GET"])
    def aa():
        return jsonify(
            {
                "setosa": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                "versicolor": {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4},
                "virginica": {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
            }
        )

    @a.route("/predict", methods=["POST"])
    def ab():
        ac = request.get_json(silent=True)
        if not ac:
            return jsonify({"error": "send json body"}), 400

        ad = []
        for ae in h:
            if ae not in ac:
                return jsonify({"error": f"missing field: {ae}"}), 400
            ad.append(float(ac[ae]))

        af = np.array(ad).reshape(1, -1)
        ag = c.transform(af)
        ah = d.predict(ag)[0]
        ai = d.predict_proba(ag)[0]
        aj = g[ah]

        ak = {
            "species": aj,
            "confidence": round(float(ai[ah]), 4),
            "probabilities": {g[al]: round(float(ai[al]), 4) for al in range(len(g))},
        }

        m(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input": {h[am]: ad[am] for am in range(len(h))},
                "result": ak,
            }
        )

        return jsonify(ak)

    print("open: http://127.0.0.1:5000")
    a.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()

import csv
import io
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, Response


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
    k = f.get("feature_ranges", {})

    def l():
        if not i.exists():
            return []
        with open(i, encoding="utf-8") as m:
            return json.load(m)

    def n(o):
        i.parent.mkdir(exist_ok=True)
        p = l()
        p.insert(0, o)
        p = p[:50]
        with open(i, "w", encoding="utf-8") as q:
            json.dump(p, q, indent=2)

    def r(s):
        t = []
        if not k:
            return t
        for u, v in s.items():
            if u not in k:
                continue
            w = k[u]
            if v < w["min"] or v > w["max"]:
                t.append(f"{u} outside trained range ({w['min']}-{w['max']})")
        return t

    @a.route("/")
    def v():
        return send_from_directory("static", "index.html")

    @a.route("/chart/<w>")
    def x(w):
        y = Path("output") / f"{w}.png"
        if not y.exists():
            return jsonify({"error": "chart not found"}), 404
        return send_from_directory("output", f"{w}.png")

    @a.route("/report")
    def ah():
        y = Path("output") / "report.html"
        if not y.exists():
            return jsonify({"error": "run report.py first"}), 404
        return send_from_directory("output", "report.html")

    @a.route("/compare", methods=["GET"])
    def ai():
        y = Path("data/compare.json")
        if not y.exists():
            return jsonify({"error": "run compare.py first"}), 404
        with open(y, encoding="utf-8") as aj:
            return jsonify(json.load(aj))

    @a.route("/health", methods=["GET"])
    def y():
        return jsonify({"status": "ok", "model": f["best_model"], "accuracy": f["accuracy"]})

    @a.route("/dataset", methods=["GET"])
    def z():
        if not j.exists():
            return jsonify({"error": "run expand_data.py first"}), 404
        with open(j, encoding="utf-8") as aa:
            return jsonify(json.load(aa))

    @a.route("/metrics", methods=["GET"])
    def ab():
        return jsonify(f)

    @a.route("/ranges", methods=["GET"])
    def ac():
        return jsonify(k)

    @a.route("/history", methods=["GET"])
    def ad():
        return jsonify(l())

    @a.route("/history", methods=["DELETE"])
    def ae():
        with open(i, "w", encoding="utf-8") as af:
            json.dump([], af)
        return jsonify({"status": "cleared"})

    @a.route("/export/history", methods=["GET"])
    def ag():
        ah = l()
        ai = io.StringIO()
        aj = csv.writer(ai)
        aj.writerow(["time", "species", "confidence", "sepal_length", "sepal_width", "petal_length", "petal_width"])
        for ak in ah:
            al = ak["result"]
            am = ak["input"]
            aj.writerow([
                ak["time"],
                al["species"],
                al["confidence"],
                am.get("sepal_length", ""),
                am.get("sepal_width", ""),
                am.get("petal_length", ""),
                am.get("petal_width", ""),
            ])
        return Response(
            ai.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    @a.route("/samples", methods=["GET"])
    def an():
        return jsonify(
            {
                "setosa": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                "versicolor": {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4},
                "virginica": {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
            }
        )

    @a.route("/predict", methods=["POST"])
    def ao():
        ap = request.get_json(silent=True)
        if not ap:
            return jsonify({"error": "send json body"}), 400

        aq = []
        for ar in h:
            if ar not in ap:
                return jsonify({"error": f"missing field: {ar}"}), 400
            aq.append(float(ap[ar]))

        as_ = {h[at]: aq[at] for at in range(len(h))}
        au = r(as_)
        av = np.array(aq).reshape(1, -1)
        aw = c.transform(av)
        ax = d.predict(aw)[0]
        ay = d.predict_proba(aw)[0]
        az = g[ax]

        ba = {
            "species": az,
            "confidence": round(float(ay[ax]), 4),
            "probabilities": {g[bb]: round(float(ay[bb]), 4) for bb in range(len(g))},
            "warnings": au,
        }

        n(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input": as_,
                "result": ba,
            }
        )

        return jsonify(ba)

    @a.route("/predict/batch", methods=["POST"])
    def ak():
        al = request.get_json(silent=True)
        if not al or "rows" not in al:
            return jsonify({"error": "send json with rows array"}), 400

        am = []
        for an, ao in enumerate(al["rows"]):
            ap = []
            for aq in h:
                if aq not in ao:
                    return jsonify({"error": f"row {an} missing {aq}"}), 400
                ap.append(float(ao[aq]))

            ar = {h[as_]: ap[as_] for as_ in range(len(h))}
            at = np.array(ap).reshape(1, -1)
            au = c.transform(at)
            av = d.predict(au)[0]
            aw = d.predict_proba(au)[0]
            ax = g[av]
            am.append(
                {
                    "species": ax,
                    "confidence": round(float(aw[av]), 4),
                    "probabilities": {g[ay]: round(float(aw[ay]), 4) for ay in range(len(g))},
                    "warnings": r(ar),
                }
            )

        return jsonify({"count": len(am), "results": am})

    print("open: http://127.0.0.1:5000")
    a.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()

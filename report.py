import json
from datetime import datetime
from pathlib import Path


def main():
    a = Path("models/metrics.json")
    b = Path("data/compare.json")
    c = Path("data/stats.json")

    if not a.exists():
        print("run train.py first")
        return

    with open(a, encoding="utf-8") as d:
        e = json.load(d)

    f = {}
    if b.exists():
        with open(b, encoding="utf-8") as g:
            f = json.load(g)

    h = {}
    if c.exists():
        with open(c, encoding="utf-8") as i:
            h = json.load(i)

    j = Path("output")
    j.mkdir(exist_ok=True)
    k = j / "report.html"

    l = ""
    for m, n in e.get("model_comparison", {}).items():
        o = "best" if m == e["best_model"] else ""
        l += f'<tr class="{o}"><td>{m}</td><td>{n["cv_mean"]*100:.1f}%</td><td>±{n["cv_std"]*100:.1f}%</td></tr>'

    p = ""
    for q, r in e.get("feature_importance", {}).items():
        p += f"<tr><td>{q}</td><td>{r:.4f}</td></tr>"

    s = ""
    if f:
        s = f"""
        <h2>original vs expanded</h2>
        <table>
          <tr><th>dataset</th><th>samples</th><th>test accuracy</th></tr>
          <tr><td>original only</td><td>{f["original_only"]["samples"]}</td><td>{f["original_only"]["test_accuracy"]*100:.1f}%</td></tr>
          <tr><td>expanded</td><td>{f["expanded"]["samples"]}</td><td>{f["expanded"]["test_accuracy"]*100:.1f}%</td></tr>
        </table>
        """

    t = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Iris ML Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #0f1419; color: #e8edf5; padding: 2rem; max-width: 900px; margin: auto; }}
    h1 {{ color: #3ecf8e; }}
    h2 {{ color: #8b9cb3; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2rem; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 0.5rem; }}
    th, td {{ border-bottom: 1px solid #2d3a4f; padding: 0.5rem; text-align: left; }}
    th {{ color: #8b9cb3; font-size: 0.8rem; }}
    tr.best td {{ color: #3ecf8e; font-weight: 600; }}
    .meta {{ color: #8b9cb3; font-size: 0.9rem; }}
    img {{ max-width: 100%; border-radius: 8px; margin-top: 0.5rem; background: #fff; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  </style>
</head>
<body>
  <h1>Iris Species Classifier — Report</h1>
  <p class="meta">generated {datetime.now().strftime("%Y-%m-%d %H:%M")} | model: {e["best_model"]} | trained: {e.get("trained_at", "n/a")}</p>

  <h2>summary</h2>
  <table>
    <tr><th>metric</th><th>value</th></tr>
    <tr><td>test accuracy</td><td>{e["accuracy"]*100:.1f}%</td></tr>
    <tr><td>val accuracy</td><td>{e.get("val_accuracy", e["accuracy"])*100:.1f}%</td></tr>
    <tr><td>f1 macro</td><td>{e.get("f1_macro", 0)*100:.1f}%</td></tr>
    <tr><td>original samples</td><td>{e.get("original_count", h.get("original_count", "?"))}</td></tr>
    <tr><td>expanded samples</td><td>{e.get("expanded_count", h.get("expanded_count", "?"))}</td></tr>
    <tr><td>train / val / test</td><td>{e.get("train_count", "?")} / {e.get("val_count", "?")} / {e.get("test_count", "?")}</td></tr>
  </table>

  {s}

  <h2>model comparison</h2>
  <table>
    <tr><th>model</th><th>cv mean</th><th>cv std</th></tr>
    {l}
  </table>

  <h2>feature importance</h2>
  <table>
    <tr><th>feature</th><th>score</th></tr>
    {p}
  </table>

  <h2>charts</h2>
  <div class="grid">
    <div><p>confusion matrix</p><img src="confusion_matrix.png" alt="confusion matrix"></div>
    <div><p>model comparison</p><img src="model_comparison.png" alt="model comparison"></div>
    <div><p>dataset scatter</p><img src="dataset_scatter.png" alt="scatter"></div>
    <div><p>class distribution</p><img src="class_distribution.png" alt="distribution"></div>
  </div>
</body>
</html>"""

    with open(k, "w", encoding="utf-8") as u:
        u.write(t)

    print(f"saved to {k.resolve()}")


if __name__ == "__main__":
    main()

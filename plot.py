import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    a = Path("models") / "metrics.json"

    if not a.exists():
        print("run train.py first")
        return

    with open(a, encoding="utf-8") as b:
        c = json.load(b)

    d = Path("output")
    d.mkdir(exist_ok=True)

    e = np.array(c["confusion_matrix"])
    f = c["classes"]

    g, h = plt.subplots(figsize=(6, 5))
    i = h.imshow(e, cmap="Blues")
    h.set_xticks(range(len(f)))
    h.set_yticks(range(len(f)))
    h.set_xticklabels(f)
    h.set_yticklabels(f)
    h.set_xlabel("predicted")
    h.set_ylabel("actual")
    h.set_title("confusion matrix")

    for j in range(e.shape[0]):
        for k in range(e.shape[1]):
            h.text(k, j, str(e[j, k]), ha="center", va="center", color="black")

    g.colorbar(i)
    g.tight_layout()
    l = d / "confusion_matrix.png"
    g.savefig(l, dpi=150)
    plt.close(g)

    m = list(c["model_comparison"].keys())
    n = [c["model_comparison"][o]["cv_mean"] for o in m]

    p, q = plt.subplots(figsize=(8, 4))
    r = q.barh(m, n, color="#5b9fd4")
    q.set_xlim(0, 1.05)
    q.set_xlabel("cv score")
    q.set_title("model comparison")
    for s, t in zip(r, n):
        q.text(t + 0.01, s.get_y() + s.get_height() / 2, f"{t:.3f}", va="center", fontsize=9)
    p.tight_layout()
    u = d / "model_comparison.png"
    p.savefig(u, dpi=150)
    plt.close(p)

    if c.get("feature_importance"):
        v = list(c["feature_importance"].keys())
        w = list(c["feature_importance"].values())
        x, y = plt.subplots(figsize=(7, 4))
        z = y.bar(range(len(v)), w, color="#3ecf8e")
        y.set_xticks(range(len(v)))
        y.set_xticklabels([a.replace(" (cm)", "") for a in v], rotation=15, ha="right")
        y.set_ylabel("importance")
        y.set_title("feature importance")
        x.tight_layout()
        aa = d / "feature_importance.png"
        x.savefig(aa, dpi=150)
        plt.close(x)
        print(f"saved to {aa.resolve()}")

    print(f"saved to {l.resolve()}")
    print(f"saved to {u.resolve()}")


if __name__ == "__main__":
    main()

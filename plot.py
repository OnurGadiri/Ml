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

    d = np.array(c["confusion_matrix"])
    e = c["classes"]

    f, g = plt.subplots(figsize=(6, 5))
    h = g.imshow(d, cmap="Blues")
    g.set_xticks(range(len(e)))
    g.set_yticks(range(len(e)))
    g.set_xticklabels(e)
    g.set_yticklabels(e)
    g.set_xlabel("predicted")
    g.set_ylabel("actual")
    g.set_title("confusion matrix")

    for m in range(d.shape[0]):
        for n in range(d.shape[1]):
            g.text(n, m, str(d[m, n]), ha="center", va="center", color="black")

    f.colorbar(h)
    f.tight_layout()

    k = Path("output")
    k.mkdir(exist_ok=True)
    l = k / "confusion_matrix.png"
    f.savefig(l, dpi=150)
    plt.close(f)

    print(f"saved to {l.resolve()}")


if __name__ == "__main__":
    main()

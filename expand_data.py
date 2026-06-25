import csv
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris


def main():
    a = load_iris()
    b = a.data
    c = a.target
    d = a.target_names
    e = list(a.feature_names)

    f = Path("data")
    f.mkdir(exist_ok=True)

    g = f / "iris_original.csv"
    h = f / "iris_expanded.csv"
    i = f / "stats.json"

    j = ["species", "source"] + e

    with open(g, "w", newline="", encoding="utf-8") as k:
        l = csv.writer(k)
        l.writerow(j)
        for m in range(b.shape[0]):
            n = [str(d[c[m]]), "original"] + [round(float(x), 4) for x in b[m]]
            l.writerow(n)

    o = np.std(b, axis=0) * 0.06
    p = 9
    q = []

    with open(h, "w", newline="", encoding="utf-8") as r:
        s = csv.writer(r)
        s.writerow(j)
        for m in range(b.shape[0]):
            t = [str(d[c[m]]), "original"] + [round(float(x), 4) for x in b[m]]
            s.writerow(t)
            q.append(t)
            for u in range(p):
                v = b[m] + np.random.default_rng(m * 100 + u).normal(0, o)
                w = [str(d[c[m]]), "augmented"] + [round(float(x), 4) for x in v]
                s.writerow(w)
                q.append(w)

    x = {
        "original_count": int(b.shape[0]),
        "expanded_count": len(q),
        "augment_factor": p + 1,
        "classes": d.tolist(),
        "features": e,
        "noise_scale": 0.06,
    }

    with open(i, "w", encoding="utf-8") as y:
        json.dump(x, y, indent=2)

    print(f"original: {x['original_count']}")
    print(f"expanded: {x['expanded_count']}")
    print(f"saved to {h.resolve()}")


if __name__ == "__main__":
    main()

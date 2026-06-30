import csv
import json
import sys
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
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 9

    with open(g, "w", newline="", encoding="utf-8") as l:
        m = csv.writer(l)
        m.writerow(j)
        for n in range(b.shape[0]):
            o = [str(d[c[n]]), "original"] + [round(float(x), 4) for x in b[n]]
            m.writerow(o)

    p = np.std(b, axis=0) * 0.06
    q = []

    with open(h, "w", newline="", encoding="utf-8") as r:
        s = csv.writer(r)
        s.writerow(j)
        for n in range(b.shape[0]):
            t = [str(d[c[n]]), "original"] + [round(float(x), 4) for x in b[n]]
            s.writerow(t)
            q.append(t)
            for u in range(k):
                v = b[n] + np.random.default_rng(n * 100 + u).normal(0, p)
                w = [str(d[c[n]]), "augmented"] + [round(float(x), 4) for x in v]
                s.writerow(w)
                q.append(w)

    x = {
        "original_count": int(b.shape[0]),
        "expanded_count": len(q),
        "augment_factor": k + 1,
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

import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else "data/samples.csv"
    b = Path(a)

    if not b.exists():
        print(f"file not found: {b}")
        sys.exit(1)

    c = Path("models")
    d = joblib.load(c / "scaler.pkl")
    e = joblib.load(c / "classifier.pkl")

    with open(c / "metrics.json", encoding="utf-8") as f:
        g = json.load(f)
    h = g["classes"]
    i = g["features"]

    j = []
    with open(b, newline="", encoding="utf-8") as k:
        l = csv.DictReader(k)
        for m in l:
            n = [float(m[x]) for x in i]
            j.append(n)

    if not j:
        print("no rows found")
        return

    o = np.array(j)
    p = d.transform(o)
    q = e.predict(p)
    r = e.predict_proba(p)

    print(f"file: {b}")
    print(f"rows: {len(j)}")
    print()
    for s, t, u in zip(j, q, r):
        v = h[t]
        w = float(u[t])
        x = ", ".join(f"{i[y]}={s[y]}" for y in range(len(i)))
        print(f"{x} -> {v} ({w:.4f})")


if __name__ == "__main__":
    main()

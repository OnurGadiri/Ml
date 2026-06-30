import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np


def main():
    a = Path("models")
    b = a / "classifier.pkl"
    c = a / "scaler.pkl"
    d = a / "metrics.json"

    if not b.exists() or not c.exists() or not d.exists():
        print("run train.py first")
        sys.exit(1)

    e = joblib.load(c)
    f = joblib.load(b)

    with open(d, encoding="utf-8") as g:
        h = json.load(g)

    i = np.array([[5.1, 3.5, 1.4, 0.2]])
    j = e.transform(i)
    k = f.predict(j)[0]
    l = h["classes"][k]
    if l != "setosa":
        print("fail: setosa sample")
        sys.exit(1)

    m = np.array([[6.3, 3.3, 6.0, 2.5]])
    n = e.transform(m)
    o = f.predict(n)[0]
    p = h["classes"][o]
    if p != "virginica":
        print("fail: virginica sample")
        sys.exit(1)

    q = subprocess.run(
        [sys.executable, "predict.py", "5.1", "3.5", "1.4", "0.2"],
        capture_output=True,
        text=True,
    )
    if q.returncode != 0 or "setosa" not in q.stdout:
        print("fail: predict.py")
        sys.exit(1)

    if h["accuracy"] < 0.9:
        print("fail: accuracy below 0.9")
        sys.exit(1)

    r = Path("data/compare.json")
    if not r.exists():
        print("fail: compare.json missing")
        sys.exit(1)

    s = Path("output/report.html")
    if not s.exists():
        print("fail: report.html missing")
        sys.exit(1)

    print("all tests passed")


if __name__ == "__main__":
    main()

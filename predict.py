import json
import sys
from pathlib import Path

import joblib
import numpy as np


def main():
    if len(sys.argv) < 5:
        print("usage: python predict.py <sepal_length> <sepal_width> <petal_length> <petal_width>")
        sys.exit(1)

    a = [float(x) for x in sys.argv[1:5]]
    b = np.array(a).reshape(1, -1)

    c = Path("models")
    d = joblib.load(c / "scaler.pkl")
    e = joblib.load(c / "classifier.pkl")

    f = d.transform(b)
    g = e.predict(f)[0]
    h = e.predict_proba(f)[0]

    with open(c / "metrics.json", encoding="utf-8") as i:
        j = json.load(i)
    k = j["classes"]

    l = k[g]

    print(f"prediction: {l}")
    for m, n in enumerate(h):
        print(f"  {k[m]}: {n:.4f}")


if __name__ == "__main__":
    main()

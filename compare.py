import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    a = Path("data/iris_original.csv")
    b = Path("models/metrics.json")

    if not a.exists():
        print("run expand_data.py first")
        return

    if not b.exists():
        print("run train.py first")
        return

    c = []
    d = []
    e = []

    with open(a, newline="", encoding="utf-8") as f:
        g = csv.DictReader(f)
        h = [x for x in g.fieldnames if "(cm)" in x]
        for i in g:
            c.append([float(i[x]) for x in h])
            d.append(i["species"])

    j = np.array(c)
    k = np.array(d)

    l = LabelEncoder()
    m = l.fit_transform(k)

    n, o, p, q = train_test_split(j, m, test_size=0.2, random_state=42, stratify=m)

    r = StandardScaler()
    s = r.fit_transform(n)
    t = r.transform(o)

    u = KNeighborsClassifier(n_neighbors=3)
    v = cross_val_score(u, s, p, cv=5)
    u.fit(s, p)
    w = u.predict(t)
    x = accuracy_score(q, w)

    with open(b, encoding="utf-8") as y:
        z = json.load(y)

    aa = {
        "original_only": {
            "samples": int(j.shape[0]),
            "cv_mean": round(float(v.mean()), 4),
            "cv_std": round(float(v.std()), 4),
            "test_accuracy": round(float(x), 4),
            "model": "k_neighbors",
        },
        "expanded": {
            "samples": z["expanded_count"],
            "test_accuracy": z["accuracy"],
            "val_accuracy": z.get("val_accuracy", z["accuracy"]),
            "best_model": z["best_model"],
            "f1_macro": z.get("f1_macro", 0),
        },
        "gain": round(z["accuracy"] - float(x), 4),
    }

    ab = Path("data/compare.json")
    with open(ab, "w", encoding="utf-8") as ac:
        json.dump(aa, ac, indent=2)

    print(f"original test accuracy: {x:.4f}")
    print(f"expanded test accuracy: {z['accuracy']:.4f}")
    print(f"saved to {ab.resolve()}")


if __name__ == "__main__":
    main()

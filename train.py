import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def main():
    a = load_iris()
    b = a.data
    c = a.target
    d = a.target_names
    e = list(a.feature_names)

    f, g, h, i = train_test_split(b, c, test_size=0.25, random_state=42, stratify=c)

    j = StandardScaler()
    k = j.fit_transform(f)
    l = j.transform(g)

    m = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=300, random_state=42),
        "k_neighbors": KNeighborsClassifier(n_neighbors=5),
    }

    n = {}
    o = None
    p = -1.0

    for q, r in m.items():
        s = cross_val_score(r, k, h, cv=5)
        t = round(float(s.mean()), 4)
        u = round(float(s.std()), 4)
        n[q] = {"cv_mean": t, "cv_std": u}
        if t > p:
            p = t
            o = q

    v = m[o]
    v.fit(k, h)

    w = v.predict(l)
    x = accuracy_score(i, w)
    y = classification_report(i, w, target_names=d, output_dict=True)
    ae = confusion_matrix(i, w).tolist()

    z = Path("models")
    z.mkdir(exist_ok=True)

    joblib.dump(v, z / "classifier.pkl")
    joblib.dump(j, z / "scaler.pkl")

    aa = {
        "best_model": o,
        "model_comparison": n,
        "accuracy": round(x, 4),
        "classes": d.tolist(),
        "features": e,
        "feature_count": int(b.shape[1]),
        "sample_count": int(b.shape[0]),
        "report": y,
        "confusion_matrix": ae,
    }

    with open(z / "metrics.json", "w", encoding="utf-8") as ab:
        json.dump(aa, ab, indent=2)

    print(f"best model: {o}")
    print(f"test accuracy: {x:.4f}")
    for ac, ad in n.items():
        print(f"  {ac}: {ad['cv_mean']:.4f} (+/- {ad['cv_std']:.4f})")
    print(f"saved to {z.resolve()}")


if __name__ == "__main__":
    main()

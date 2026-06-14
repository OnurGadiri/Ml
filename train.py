import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    a = load_iris()
    b = a.data
    c = a.target
    d = a.target_names

    e, f, g, h = train_test_split(b, c, test_size=0.25, random_state=42, stratify=c)

    i = StandardScaler()
    j = i.fit_transform(e)
    k = i.transform(f)

    l = RandomForestClassifier(n_estimators=100, random_state=42)
    l.fit(j, g)

    m = l.predict(k)
    n = accuracy_score(h, m)
    o = classification_report(h, m, target_names=d, output_dict=True)

    p = Path("models")
    p.mkdir(exist_ok=True)

    joblib.dump(l, p / "classifier.pkl")
    joblib.dump(i, p / "scaler.pkl")

    q = {
        "accuracy": round(n, 4),
        "classes": d.tolist(),
        "feature_count": int(b.shape[1]),
        "sample_count": int(b.shape[0]),
        "report": o,
    }

    with open(p / "metrics.json", "w", encoding="utf-8") as r:
        json.dump(q, r, indent=2)

    print(f"accuracy: {n:.4f}")
    print(f"saved to {p.resolve()}")


if __name__ == "__main__":
    main()

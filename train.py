import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def main():
    a = Path("data/iris_expanded.csv")
    if not a.exists():
        print("run expand_data.py first")
        return

    b = []
    c = []
    m = []

    with open(a, newline="", encoding="utf-8") as f:
        g = csv.DictReader(f)
        m = [x for x in g.fieldnames if "(cm)" in x]
        for h in g:
            i = [float(h[x]) for x in m]
            b.append(i)
            c.append(h["species"])

    j = np.array(b)
    k = np.array(c)

    n = LabelEncoder()
    o = n.fit_transform(k)

    p, q, r, s = train_test_split(j, o, test_size=0.2, random_state=42, stratify=o)
    t, u, v, w = train_test_split(p, r, test_size=0.25, random_state=42, stratify=r)

    x = StandardScaler()
    y = x.fit_transform(t)
    z = x.transform(u)
    aa = x.transform(q)

    ab = {
        "random_forest": RandomForestClassifier(random_state=42),
        "logistic_regression": LogisticRegression(max_iter=500, random_state=42),
        "k_neighbors": KNeighborsClassifier(),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "svm": SVC(probability=True, random_state=42),
    }

    ac = {}
    ad = None
    ae = -1.0

    for af, ag in ab.items():
        ah = cross_val_score(ag, y, v, cv=5)
        ai = round(float(ah.mean()), 4)
        aj = round(float(ah.std()), 4)
        ac[af] = {"cv_mean": ai, "cv_std": aj}
        if ai > ae:
            ae = ai
            ad = af

    ak = {
        "random_forest": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, None]},
        "logistic_regression": {"C": [0.1, 1, 10]},
        "k_neighbors": {"n_neighbors": [3, 5, 7, 11]},
        "gradient_boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        "decision_tree": {"max_depth": [3, 5, 8, None]},
        "svm": {"C": [0.5, 1, 2], "kernel": ["rbf", "linear"]},
    }

    al = GridSearchCV(ab[ad], ak[ad], cv=5, n_jobs=-1)
    al.fit(y, v)
    am = al.best_estimator_
    am.fit(y, v)

    an = am.predict(z)
    ao = am.predict(aa)
    ap = accuracy_score(w, an)
    aq = accuracy_score(s, ao)
    ar = classification_report(s, ao, target_names=n.classes_, output_dict=True)
    as_ = confusion_matrix(s, ao).tolist()

    at = {}
    if hasattr(am, "feature_importances_"):
        for au, av in zip(m, am.feature_importances_):
            at[au] = round(float(av), 4)
    elif hasattr(am, "coef_"):
        aw = np.abs(am.coef_).mean(axis=0)
        for au, av in zip(m, aw):
            at[au] = round(float(av), 4)
    else:
        ay = mutual_info_classif(y, v, random_state=42)
        for au, av in zip(m, ay):
            at[au] = round(float(av), 4)

    bg = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    bf = {}
    for bh, bi in enumerate(bg):
        bf[bi] = {
            "min": round(float(t[:, bh].min()), 2),
            "max": round(float(t[:, bh].max()), 2),
            "mean": round(float(t[:, bh].mean()), 2),
        }

    ax = Path("models")
    ax.mkdir(exist_ok=True)

    joblib.dump(am, ax / "classifier.pkl")
    joblib.dump(x, ax / "scaler.pkl")
    joblib.dump(n, ax / "labels.pkl")

    ay = Path("data/stats.json")
    with open(ay, encoding="utf-8") as az:
        ba = json.load(az)

    bb = {
        "best_model": ad,
        "tuned_params": al.best_params_,
        "model_comparison": ac,
        "val_accuracy": round(ap, 4),
        "accuracy": round(aq, 4),
        "classes": n.classes_.tolist(),
        "features": m,
        "feature_keys": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "feature_importance": at,
        "feature_ranges": bf,
        "feature_count": int(j.shape[1]),
        "sample_count": int(j.shape[0]),
        "original_count": ba["original_count"],
        "expanded_count": ba["expanded_count"],
        "train_count": int(t.shape[0]),
        "val_count": int(u.shape[0]),
        "test_count": int(q.shape[0]),
        "report": ar,
        "confusion_matrix": as_,
    }

    with open(ax / "metrics.json", "w", encoding="utf-8") as bc:
        json.dump(bb, bc, indent=2)

    print(f"best model: {ad}")
    print(f"tuned: {al.best_params_}")
    print(f"val accuracy: {ap:.4f}")
    print(f"test accuracy: {aq:.4f}")
    for bd, be in ac.items():
        print(f"  {bd}: {be['cv_mean']:.4f} (+/- {be['cv_std']:.4f})")
    print(f"saved to {ax.resolve()}")


if __name__ == "__main__":
    main()

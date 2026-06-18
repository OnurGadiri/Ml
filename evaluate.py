import json
from pathlib import Path


def main():
    a = Path("models") / "metrics.json"

    if not a.exists():
        print("run train.py first")
        return

    with open(a, encoding="utf-8") as b:
        c = json.load(b)

    print(f"best model: {c['best_model']}")
    print(f"test accuracy: {c['accuracy']}")
    print(f"samples: {c['sample_count']}")
    print(f"features: {', '.join(c['features'])}")
    print()
    print("model comparison (5-fold cv):")
    for d, e in c["model_comparison"].items():
        print(f"  {d}: {e['cv_mean']:.4f} (+/- {e['cv_std']:.4f})")
    print()
    print("per-class scores:")
    for f in c["classes"]:
        g = c["report"][f]
        print(f"  {f}: precision={g['precision']:.3f} recall={g['recall']:.3f} f1={g['f1-score']:.3f}")
    print()
    print("confusion matrix:")
    h = c["classes"]
    i = c["confusion_matrix"]
    j = "     " + "  ".join(f"{x:>5}" for x in h)
    print(j)
    for k, l in enumerate(i):
        m = "  ".join(f"{x:>5}" for x in l)
        print(f"{h[k]:>5} {m}")


if __name__ == "__main__":
    main()

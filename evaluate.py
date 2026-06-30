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
    print(f"tuned params: {c.get('tuned_params', {})}")
    print(f"val accuracy: {c.get('val_accuracy', c['accuracy'])}")
    print(f"test accuracy: {c['accuracy']}")
    print(f"f1 macro: {c.get('f1_macro', c['accuracy'])}")
    print(f"trained at: {c.get('trained_at', 'n/a')}")
    print(f"original samples: {c.get('original_count', c['sample_count'])}")
    print(f"expanded samples: {c.get('expanded_count', c['sample_count'])}")
    print(f"train / val / test: {c.get('train_count', '?')} / {c.get('val_count', '?')} / {c.get('test_count', '?')}")
    print(f"features: {', '.join(c['features'])}")
    print()
    print("model comparison (5-fold cv):")
    for d, e in c["model_comparison"].items():
        print(f"  {d}: {e['cv_mean']:.4f} (+/- {e['cv_std']:.4f})")
    print()
    if c.get("feature_importance"):
        print("feature importance:")
        for f, g in c["feature_importance"].items():
            print(f"  {f}: {g:.4f}")
        print()
    if c.get("feature_ranges"):
        print("feature ranges (train):")
        for h, i in c["feature_ranges"].items():
            print(f"  {h}: {i['min']} - {i['max']} (mean {i['mean']})")
        print()
    print("per-class scores:")
    for h in c["classes"]:
        i = c["report"][h]
        print(f"  {h}: precision={i['precision']:.3f} recall={i['recall']:.3f} f1={i['f1-score']:.3f}")
    print()
    print("confusion matrix:")
    j = c["classes"]
    k = c["confusion_matrix"]
    l = "     " + "  ".join(f"{x:>5}" for x in j)
    print(l)
    for m, n in enumerate(k):
        o = "  ".join(f"{x:>5}" for x in n)
        print(f"{j[m]:>5} {o}")


if __name__ == "__main__":
    main()

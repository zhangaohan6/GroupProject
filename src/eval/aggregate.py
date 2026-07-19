"""[A] Collate all results/*.json into the report comparison table — CONTRACTS §5.

  python -m src.eval.aggregate --results results
"""
import argparse, glob, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rows = []
    for rj in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        r = json.load(open(rj))
        if r.get("degradation"):
            continue  # clean runs only in the headline table
        m, t = r["metrics"], r["timing"]
        rows.append({"run": r["run"], "top1": m["top1"], "top5": m["top5"],
                     "macro_f1": m["macro_f1"], "bal_acc": m["balanced_accuracy"],
                     "train_s": t["train_seconds"], "predict_s": t["predict_seconds"]})
    if not rows:
        print("No clean results/*.json found."); return
    cols = ["run", "top1", "top5", "macro_f1", "bal_acc", "train_s", "predict_s"]
    def fmt(v): return f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))
    w = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(w[c]) for c in cols))
    print("-+-".join("-" * w[c] for c in cols))
    for r in sorted(rows, key=lambda d: -(d["macro_f1"] or 0)):
        print(" | ".join(fmt(r[c]).ljust(w[c]) for c in cols))


if __name__ == "__main__":
    main()

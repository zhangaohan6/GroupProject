"""[A] Aggregate all runs into the comparison table the report needs, and run the
data-leakage check. Each method's runner already wrote results/<run_id>/result.json
via common.save_result; this just collates + verifies.

  python evaluate.py --results ../results                 # comparison table
  python evaluate.py --data ../data --leakage_check       # assert no split overlap
"""
import argparse, glob, json, os
from common import read_manifest


def table(results_dir):
    rows = []
    for rj in sorted(glob.glob(os.path.join(results_dir, "*", "result.json"))):
        r = json.load(open(rj))
        rows.append(r)
    if not rows:
        print("No result.json found under", results_dir); return
    cols = ["method", "top1", "top5", "macro_f1", "balanced_acc",
            "train_seconds", "test_seconds"]
    w = {c: max(len(c), *(len(f"{r.get(c):.3f}" if isinstance(r.get(c), float)
                          else str(r.get(c))) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(w[c]) for c in cols))
    print("-+-".join("-" * w[c] for c in cols))
    for r in sorted(rows, key=lambda d: -(d.get("macro_f1") or 0)):
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append((f"{v:.3f}" if isinstance(v, float) else str(v)).ljust(w[c]))
        print(" | ".join(cells))


def leakage_check(data_root):
    sets = {}
    for s in ["train", "val", "test"]:
        rows = read_manifest(os.path.join(data_root, "manifests", f"{s}.csv"))
        sets[s] = {r["path"] for r in rows}
    ok = True
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = sets[a] & sets[b]
        print(f"{a}∩{b} = {len(inter)}  {'OK' if not inter else 'LEAK!'}")
        ok &= not inter
    print("Sizes:", {k: len(v) for k, v in sets.items()})
    print("LEAKAGE CHECK", "PASSED" if ok else "FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--leakage_check", action="store_true")
    args = ap.parse_args()
    if args.leakage_check:
        leakage_check(args.data)
    else:
        table(args.results)


if __name__ == "__main__":
    main()

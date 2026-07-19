#!/usr/bin/env python3
"""Build the graded subset from iNaturalist-2021 mini.

Samples SEED-fixed N_CLASSES species, splits train_mini into train/val
(40/10 per class by default) and copies N_TEST val images per class as
the held-out test set. Writes a manifest so the exact subset is reproducible.

Usage:
  python build_subset.py \
      --inat_root /path/to/inat2021 \
      --out ../data/subset500 \
      --n_classes 500 --n_train 40 --n_val 10 --n_test 10 --seed 42

Expects under --inat_root:
  train_mini/                 (extracted images)
  train_mini.json             (annotations)
  val/                        (extracted images)
  val.json                    (annotations)
"""
import argparse, json, os, random, shutil
from collections import defaultdict


def load_coco(json_path):
    with open(json_path) as f:
        d = json.load(f)
    # image_id -> file_name ; image_id -> category_id
    fn = {im["id"]: im["file_name"] for im in d["images"]}
    cat = {a["image_id"]: a["category_id"] for a in d["annotations"]}
    by_cat = defaultdict(list)  # category_id -> [file_name,...]
    for img_id, c in cat.items():
        by_cat[c].append(fn[img_id])
    cats = {c["id"]: c["name"] for c in d["categories"]}
    return by_cat, cats


def link_or_copy(src, dst, mode):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "symlink":
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inat_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_classes", type=int, default=500)
    ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--n_val", type=int, default=10)
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                    help="symlink saves disk; copy for portability/Colab")
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    print("Loading annotations ...")
    tr_by_cat, cats = load_coco(os.path.join(args.inat_root, "train_mini.json"))
    va_by_cat, _ = load_coco(os.path.join(args.inat_root, "val.json"))

    # sample classes that have enough images in BOTH splits
    eligible = [c for c in tr_by_cat
                if len(tr_by_cat[c]) >= args.n_train + args.n_val
                and len(va_by_cat.get(c, [])) >= args.n_test]
    if len(eligible) < args.n_classes:
        raise SystemExit(f"Only {len(eligible)} eligible classes, need {args.n_classes}")
    chosen = sorted(rnd.sample(eligible, args.n_classes))

    manifest = {"seed": args.seed, "n_classes": args.n_classes,
                "n_train": args.n_train, "n_val": args.n_val,
                "n_test": args.n_test, "classes": {}}

    for new_idx, cid in enumerate(chosen):
        tr = tr_by_cat[cid][:]; rnd.shuffle(tr)
        va = va_by_cat[cid][:]; rnd.shuffle(va)
        train = tr[:args.n_train]
        val = tr[args.n_train:args.n_train + args.n_val]
        test = va[:args.n_test]
        label = f"{new_idx:04d}"
        for split, files, root in [("train", train, "train_mini"),
                                    ("val", val, "train_mini"),
                                    ("test", test, "val")]:
            for fpath in files:
                src = os.path.join(args.inat_root, root, fpath)
                dst = os.path.join(args.out, split, label, os.path.basename(fpath))
                link_or_copy(src, dst, args.mode)
        manifest["classes"][label] = {"category_id": cid, "name": cats.get(cid, "")}
        if new_idx % 50 == 0:
            print(f"  {new_idx}/{args.n_classes} classes done")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. Subset at {args.out} ; manifest.json written (reproducible via seed={args.seed}).")


if __name__ == "__main__":
    main()

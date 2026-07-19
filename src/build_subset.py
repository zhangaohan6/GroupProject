#!/usr/bin/env python3
"""[A / data base] Build the graded subset per CONTRACTS.md.

Freezes classes_500.json (with taxonomy), resizes images to 256px short side JPEG
under data/images/<class_id>/, and writes manifests/{train,val,test}.csv.
Reproducible via --seed. Run ONCE for the whole team; upload data/ to shared Drive.

  python build_subset.py --inat_root /path/to/inat2021 --out ../data \
      --n_classes 500 --n_train 40 --n_val 10 --n_test 10 --seed 42
Expects extracted train_mini/, val/ and train_mini.json, val.json under --inat_root.
(For streaming extraction straight from the .tar.gz to avoid 42GB on disk, see the
 note at the bottom — this version reads an already-extracted tree, which is simpler
 and robust; swap in a tarfile stream if disk is tight.)
"""
import argparse, csv, json, os, random
from collections import defaultdict
from PIL import Image


def load_coco(json_path):
    with open(json_path) as f:
        d = json.load(f)
    fn = {im["id"]: im["file_name"] for im in d["images"]}
    by_cat = defaultdict(list)
    for a in d["annotations"]:
        by_cat[a["category_id"]].append(fn[a["image_id"]])
    cats = {c["id"]: c for c in d["categories"]}  # keeps name + taxonomy fields
    return by_cat, cats


def resize_save(src, dst, short=256):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = short / min(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im.save(dst, "JPEG", quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inat_root", required=True)
    ap.add_argument("--out", default="../data")
    ap.add_argument("--n_classes", type=int, default=500)
    ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--n_val", type=int, default=10)
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--short", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    tr_by_cat, cats = load_coco(os.path.join(args.inat_root, "train_mini.json"))
    va_by_cat, _ = load_coco(os.path.join(args.inat_root, "val.json"))
    eligible = [c for c in tr_by_cat
                if len(tr_by_cat[c]) >= args.n_train + args.n_val
                and len(va_by_cat.get(c, [])) >= args.n_test]
    if len(eligible) < args.n_classes:
        raise SystemExit(f"Only {len(eligible)} eligible classes, need {args.n_classes}")
    chosen = sorted(rnd.sample(eligible, args.n_classes))

    os.makedirs(os.path.join(args.out, "manifests"), exist_ok=True)
    classes = {"seed": args.seed, "n_classes": args.n_classes, "classes": []}
    rows = {"train": [], "val": [], "test": []}

    for cid_idx, cid in enumerate(chosen):
        meta = cats.get(cid, {})
        classes["classes"].append({
            "class_id": cid_idx, "category_id": cid,
            "name": meta.get("name", ""),
            "genus": meta.get("genus", ""), "family": meta.get("family", ""),
            "kingdom": meta.get("kingdom", "")})
        tr = tr_by_cat[cid][:]; rnd.shuffle(tr)
        va = va_by_cat[cid][:]; rnd.shuffle(va)
        plan = [("train", tr[:args.n_train], "train_mini"),
                ("val", tr[args.n_train:args.n_train + args.n_val], "train_mini"),
                ("test", va[:args.n_test], "val")]
        for split, files, root in plan:
            for fp in files:
                src = os.path.join(args.inat_root, root, fp)
                rel = os.path.join("images", str(cid_idx), os.path.basename(fp))
                dst = os.path.join(args.out, rel)
                resize_save(src, dst, args.short)
                rows[split].append({"path": rel, "class_id": cid_idx, "split": split})
        if cid_idx % 50 == 0:
            print(f"  {cid_idx}/{args.n_classes} classes")

    with open(os.path.join(args.out, "classes_500.json"), "w") as f:
        json.dump(classes, f, indent=2)
    for split, rr in rows.items():
        with open(os.path.join(args.out, "manifests", f"{split}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "class_id", "split"])
            w.writeheader(); w.writerows(rr)
    print("Done. classes_500.json + manifests/{train,val,test}.csv written.")
    print(f"  train={len(rows['train'])} val={len(rows['val'])} test={len(rows['test'])}"
          f" images under {args.out}/images/  (reproducible, seed={args.seed})")


if __name__ == "__main__":
    main()

# Streaming note: to avoid extracting 42GB, iterate `tarfile.open(...,'r|gz')` once,
# and for each member whose file_name is in your chosen set, resize+save on the fly.
# The two-pass logic above (annotations decide membership) is identical; only the
# image source changes from a path on disk to a tar member stream.

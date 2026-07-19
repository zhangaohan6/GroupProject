"""[A] Build the frozen subset — CONTRACTS §2. Freezes classes_500.json (full taxonomy),
resizes to 256px short side under images_256/<class_id 3-digit>/, writes manifests with
columns filepath,class_id,species. Reproducible via seed. Run ONCE; upload data/ to Drive.

  python -m src.data.build_subset --inat_root /path/to/inat2021 --out data \
      --n_classes 500 --n_train 40 --n_val 10 --n_test 10 --seed 42
Expects extracted train_mini/, val/ + train_mini.json, val.json under --inat_root.
"""
import argparse, csv, json, os
from collections import defaultdict
from datetime import date
from PIL import Image
from src.common.io import set_seed
import random


def load_coco(p):
    d = json.load(open(p))
    fn = {im["id"]: im["file_name"] for im in d["images"]}
    by = defaultdict(list)
    for a in d["annotations"]:
        by[a["category_id"]].append(fn[a["image_id"]])
    return by, {c["id"]: c for c in d["categories"]}


def resize_save(src, dst, short=256):
    im = Image.open(src).convert("RGB"); w, h = im.size; s = short / min(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    os.makedirs(os.path.dirname(dst), exist_ok=True); im.save(dst, "JPEG", quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inat_root", required=True); ap.add_argument("--out", default="data")
    ap.add_argument("--n_classes", type=int, default=500); ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--n_val", type=int, default=10); ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--short", type=int, default=256); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifests_only", action="store_true",
                    help="freeze classes_500.json + manifests from the JSONs only; skip image resize "
                         "(images fetched separately by fetch_images.py). Needs only the small jsons.")
    a = ap.parse_args(); set_seed(a.seed); rnd = random.Random(a.seed)

    tr_by, cats = load_coco(os.path.join(a.inat_root, "train_mini.json"))
    va_by, _ = load_coco(os.path.join(a.inat_root, "val.json"))
    elig = [c for c in tr_by if len(tr_by[c]) >= a.n_train + a.n_val and len(va_by.get(c, [])) >= a.n_test]
    if len(elig) < a.n_classes:
        raise SystemExit(f"only {len(elig)} eligible, need {a.n_classes}")
    chosen = sorted(rnd.sample(elig, a.n_classes))

    classes = {"seed": a.seed, "n_classes": a.n_classes, "source_split": "train_mini",
               "created": str(date.today()), "classes": []}
    manif = {"train": [], "val": [], "test": []}
    TAX = ["kingdom", "phylum", "class", "order", "family", "genus"]
    os.makedirs(os.path.join(a.out, "manifests"), exist_ok=True)

    for cid, inat in enumerate(chosen):
        meta = cats.get(inat, {})
        species = meta.get("name", "")
        entry = {"class_id": cid, "inat_category_id": inat,
                 "dir_name": meta.get("image_dir_name", f"{inat:05d}"),
                 **{t: meta.get(t, "") for t in TAX}, "species": species}
        classes["classes"].append(entry)
        cdir = f"{cid:03d}"
        tr = tr_by[inat][:]; rnd.shuffle(tr); va = va_by[inat][:]; rnd.shuffle(va)
        for split, files, root in [("train", tr[:a.n_train], "train_mini"),
                                   ("val", tr[a.n_train:a.n_train + a.n_val], "train_mini"),
                                   ("test", va[:a.n_test], "val")]:
            for fp in files:  # fp already includes the split prefix, e.g. "train_mini/<dir>/<img>"
                bn = os.path.basename(fp); rel = f"{cdir}/{bn}"
                if not a.manifests_only:
                    resize_save(os.path.join(a.inat_root, fp),
                                os.path.join(a.out, "images_256", rel), a.short)
                # _src is the archive member name (for fetch_images.py streaming)
                manif[split].append({"filepath": rel, "class_id": cid, "species": species, "_src": fp})
        if cid % 50 == 0:
            print(f"  {cid}/{a.n_classes}")

    json.dump(classes, open(os.path.join(a.out, "classes_500.json"), "w"), indent=2)
    fetch_map = {}  # filepath -> source path inside the tarballs (for fetch_images.py)
    for s, rows in manif.items():
        with open(os.path.join(a.out, "manifests", f"{s}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filepath", "class_id", "species"])
            w.writeheader()
            for r in rows:
                fetch_map[r["filepath"]] = r["_src"]
                w.writerow({k: r[k] for k in ["filepath", "class_id", "species"]})
    json.dump(fetch_map, open(os.path.join(a.out, "_fetch_map.json"), "w"))
    print(f"Done. classes_500.json + manifests/ (+ _fetch_map.json). "
          f"train={len(manif['train'])} val={len(manif['val'])} test={len(manif['test'])}"
          + ("  [manifests_only: images NOT fetched — run src.data.fetch_images]" if a.manifests_only else ""))


if __name__ == "__main__":
    main()

# Streaming note: to skip extracting 42GB, iterate tarfile.open(..,'r|gz') once and
# resize+save members whose file_name is in the chosen set. Membership logic unchanged.

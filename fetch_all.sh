#!/bin/bash
set -o pipefail
cd ~/Desktop/COMP9517-GroupProject
BASE="https://ml-inat-competition-datasets.s3.amazonaws.com/2021"
echo "[$(date '+%H:%M:%S')] START val.tar.gz (8.4GB, ~26min)"
curl -sS --retry 5 --retry-delay 10 --retry-all-errors "$BASE/val.tar.gz" \
  | python3 -m src.data.fetch_images --tar - --out data --map _fetch_map.json
echo "[$(date '+%H:%M:%S')] val done. START train_mini.tar.gz (42GB, ~2.2h)"
curl -sS --retry 5 --retry-delay 10 --retry-all-errors "$BASE/train_mini.tar.gz" \
  | python3 -m src.data.fetch_images --tar - --out data --map _fetch_map.json
echo "[$(date '+%H:%M:%S')] train done."
# verify counts vs manifests
python3 - <<'PY'
import os, csv, glob
have=set()
for p in glob.glob("data/images_256/*/*"):
    have.add(os.path.relpath(p,"data/images_256"))
for s in ["train","val","test"]:
    want={r["filepath"] for r in csv.DictReader(open(f"manifests/{s}.csv"))}
    miss=len(want-have)
    print(f"{s}: {len(want&have)}/{len(want)} present, missing={miss}")
print("total images:", len(have))
PY
echo "[$(date '+%H:%M:%S')] ALL DONE"

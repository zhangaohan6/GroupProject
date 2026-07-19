"""[A] Manifest read + validation — CONTRACTS §2.3, §8."""
import json
import os
import pandas as pd

COLS = ["filepath", "class_id", "species"]


def read_manifest(csv_path):
    """Returns a DataFrame with columns filepath,class_id,species (row order preserved)."""
    df = pd.read_csv(csv_path, dtype={"filepath": str, "class_id": int, "species": str})
    assert list(df.columns) == COLS, f"{csv_path} cols must be {COLS}, got {list(df.columns)}"
    return df


def load_classes(json_path):
    with open(json_path) as f:
        return json.load(f)


def validate_split(data_root, per_class=(40, 10, 10)):
    """CONTRACTS §8 leakage + count assertions. Returns (ok, report)."""
    mdir = os.path.join(data_root, "manifests")
    dfs = {s: read_manifest(os.path.join(mdir, f"{s}.csv")) for s in ["train", "val", "test"]}
    sets = {s: set(d["filepath"]) for s, d in dfs.items()}
    rep, ok = [], True
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = len(sets[a] & sets[b])
        ok &= inter == 0
        rep.append(f"{a}∩{b}={inter} {'OK' if not inter else 'LEAK!'}")
    for s, n in zip(["train", "val", "test"], per_class):
        vc = dfs[s]["class_id"].value_counts()
        good = (vc == n).all() and len(vc) == 500
        ok &= good
        rep.append(f"{s}: {len(dfs[s])} rows, {len(vc)} classes, per-class=={n}? {'OK' if good else 'BAD'}")
    return ok, "\n".join(rep)

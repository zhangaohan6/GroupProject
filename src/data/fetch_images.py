"""[A] Stream images_256/ from the big tarballs — the 42GB half of M1.

Reads _fetch_map.json (filepath -> archive member name, written by build_subset) and
streams train_mini.tar.gz / val.tar.gz ONCE each with tarfile 'r|gz', extracting + resizing
ONLY the ~30k images we need. Never lands 42GB on disk. Run on Colab (free bandwidth):

  # from a local tarball (already downloaded):
  python -m src.data.fetch_images --tar /path/train_mini.tar.gz --out data --map _fetch_map.json
  python -m src.data.fetch_images --tar /path/val.tar.gz        --out data --map _fetch_map.json
  # or stream straight from the URL (no full local copy):
  python -m src.data.fetch_images --url \
    https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz --out data
"""
import argparse, io, json, os, tarfile
from PIL import Image

BASE = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021"


def resize_save(fileobj, dst, short=256):
    im = Image.open(fileobj).convert("RGB"); w, h = im.size; s = short / min(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    os.makedirs(os.path.dirname(dst), exist_ok=True); im.save(dst, "JPEG", quality=90)


def stream(fh, want, out, short):
    """fh: file-like over the .tar.gz. want: dict member_name -> target filepath."""
    done = 0
    with tarfile.open(fileobj=fh, mode="r|gz") as tar:
        for m in tar:
            tgt = want.get(m.name)
            if tgt is None:
                continue
            resize_save(tar.extractfile(m), os.path.join(out, "images_256", tgt), short)
            done += 1
            if done % 2000 == 0:
                print(f"  extracted {done}/{len(want)}")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--map", default="_fetch_map.json")
    ap.add_argument("--tar", default="", help="local .tar.gz path")
    ap.add_argument("--url", default="", help="stream .tar.gz from URL (e.g. the S3 links)")
    ap.add_argument("--short", type=int, default=256)
    a = ap.parse_args()
    fmap = json.load(open(a.map))                       # filepath -> member name
    want = {v: k for k, v in fmap.items()}              # member name -> filepath (this archive filters itself)

    if a.tar == "-":
        import sys
        n = stream(sys.stdin.buffer, want, a.out, a.short)   # curl ... | python -m ... --tar -
    elif a.tar:
        with open(a.tar, "rb") as fh:
            n = stream(fh, want, a.out, a.short)
    elif a.url:
        import urllib.request
        with urllib.request.urlopen(a.url) as resp:
            n = stream(resp, want, a.out, a.short)
    else:
        raise SystemExit("give --tar or --url")
    print(f"Done. {n} images written under {a.out}/images_256/  "
          f"(run once per archive: train_mini.tar.gz then val.tar.gz).")


if __name__ == "__main__":
    main()

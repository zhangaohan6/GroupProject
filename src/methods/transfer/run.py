"""[D] Transfer learning (ImageNet-pretrained) runner.
  python -m src.methods.transfer.run --arch resnet50 --tag res50_ft --epochs 30
"""
import argparse, json, os, time
from src.common.io import set_seed
from src.common.manifest import read_manifest
from src.methods.deepnet import DeepMethod
from src.methods.base import save_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--tag", default="res50_ft"); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no_aug", action="store_true"); ap.add_argument("--resume", default="")
    a = ap.parse_args(); set_seed()
    tr = read_manifest(f"{a.data}/manifests/train.csv"); va = read_manifest(f"{a.data}/manifests/val.csv")
    te = read_manifest(f"{a.data}/manifests/test.csv")
    m = DeepMethod(a.arch, pretrained=True, data_root=a.data, tag=a.tag)
    ck = f"{a.data}/checkpoints/{m.name}__{a.tag}"
    m.fit(tr, va, epochs=a.epochs, lr=a.lr, bs=a.bs, augment=not a.no_aug, ckpt_dir=ck, resume=a.resume)
    os.makedirs("results", exist_ok=True)
    json.dump({"history": m.history}, open(f"results/{m.name}__{a.tag}__history.json", "w"))
    t0 = time.time(); probs = m.predict_proba(te); ps = time.time() - t0
    res = save_run(".", m.name, a.tag, "test", te["class_id"].to_numpy(), probs, m.config,
                   m.train_seconds, ps)
    print("saved", res["run"], {k: res["metrics"][k] for k in ("top1", "top5", "macro_f1")})

if __name__ == "__main__":
    main()

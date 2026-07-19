"""[E / ADVANCED #2] Robustness sweep — drives ANY method via the image_transform hook.
Model fixed, test-only, online degradation. Writes one result/prediction per (corruption,sev)
using the contract run-name (degradation tagged), so curves collate from results/.

  # deep model:
  python -m src.advanced.robustness.run --method deep --arch resnet50 \
      --ckpt data/checkpoints/transfer_resnet50__res50_ft/best.pt --tag res50_ft
  # traditional:
  python -m src.advanced.robustness.run --method bovw --bovw data/checkpoints/bovw.pkl --tag sift512
"""
import argparse, time
from src.common.io import set_seed
from src.common.manifest import read_manifest
from src.methods.base import save_run
from src.advanced.robustness.degradations import CORRUPTIONS


def load_method(args):
    if args.method == "deep":
        from src.methods.deepnet import DeepMethod
        m = DeepMethod(args.arch, pretrained=False, data_root=args.data, tag=args.tag)
        m.load(args.ckpt); return m
    from src.methods.traditional.bovw import BoVWMethod
    return BoVWMethod(args.data).load(args.bovw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--method", choices=["deep", "bovw"], required=True)
    ap.add_argument("--arch", default="resnet50"); ap.add_argument("--ckpt", default=""); ap.add_argument("--bovw", default="")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(); set_seed()
    m = load_method(a)
    te = read_manifest(f"{a.data}/manifests/test.csv"); y = te["class_id"].to_numpy()

    # clean baseline
    t0 = time.time(); probs = m.predict_proba(te); ps = time.time() - t0
    r = save_run(".", m.name, a.tag, "test", y, probs, m.config, None, ps)
    print("clean", {k: r["metrics"][k] for k in ("top1", "macro_f1")})
    for name, (make, sevs) in CORRUPTIONS.items():
        for i, s in enumerate(sevs, 1):
            probs = m.predict_proba(te, image_transform=make(s))
            r = save_run(".", m.name, a.tag, "test", y, probs, m.config, None, None,
                         degradation={"type": name, "severity": i})
            print(f"{name} s{i}", {k: r['metrics'][k] for k in ('top1', 'macro_f1')})
    print("Done. Curves: read results/*__<corruption>_s*__test.json (top1/macro_f1 vs severity).")


if __name__ == "__main__":
    main()

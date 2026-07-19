"""[C/D] Deep pipeline implementing common.Method. Trains scratch or pretrained,
logs train/val curves, checkpoints, then evaluates test via predict_proba + save_result.

  python deep_cnn.py --data ../data --arch resnet50 --pretrained --epochs 30 \
      --run_id resnet50_pretrained_42 --out ../results
  python deep_cnn.py --data ../data --arch resnet50 --epochs 60 \
      --run_id resnet50_scratch_42 --out ../results          # from scratch
Ablations: --no_aug  (augmentation off);  --arch resnet18|efficientnet_b0
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from torchvision import models
from common import Method, N_CLASSES, read_manifest, load_classes, save_result
from dataset import loader_from_manifest, ManifestDS, tensor_tf

ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def build_backbone(arch, n_classes, pretrained):
    ctor, w = ARCHS[arch]
    m = ctor(weights=w if pretrained else None)
    if arch.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    return m


class DeepMethod(Method):
    def __init__(self, arch="resnet50", pretrained=True, img=224, device=None):
        self.name = f"{arch}_{'pretrained' if pretrained else 'scratch'}"
        self.arch, self.img = arch, img
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_backbone(arch, N_CLASSES, pretrained).to(self.device)

    def fit(self, train_rows, val_rows, data_root, epochs=30, lr=1e-3, bs=64,
            augment=True, out=None, resume=""):
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        tr, _ = loader_from_manifest(data_root, "train", self.img, bs, train=True, augment=augment)
        va, _ = loader_from_manifest(data_root, "val", self.img, bs)
        start, hist, best, t0 = 0, [], 0.0, time.time()
        if resume and os.path.exists(resume):
            ck = torch.load(resume, map_location=self.device)
            self.model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            start, hist, best = ck["epoch"] + 1, ck["history"], ck["best"]
        for ep in range(start, epochs):
            trl, tra = self._epoch(tr, crit, opt)
            val, vaa = self._epoch(va, crit)
            sched.step()
            hist.append({"epoch": ep, "train_loss": trl, "train_acc": tra,
                         "val_loss": val, "val_acc": vaa})
            print(f"ep{ep:02d} train {trl:.3f}/{tra:.3f} val {val:.3f}/{vaa:.3f}")
            if out:
                ck = {"model": self.model.state_dict(), "opt": opt.state_dict(),
                      "epoch": ep, "history": hist, "best": best, "arch": self.arch}
                torch.save(ck, os.path.join(out, "last.pt"))
                if vaa > best:
                    best = vaa; torch.save(ck, os.path.join(out, "best.pt"))
        self.train_seconds = time.time() - t0
        self.history = hist
        return self

    def _epoch(self, loader, crit, opt=None):
        train = opt is not None
        self.model.train(train)
        tot = corr = 0; ls = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.set_grad_enabled(train):
                o = self.model(x); loss = crit(o, y)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            ls += loss.item() * x.size(0); corr += (o.argmax(1) == y).sum().item(); tot += x.size(0)
        return ls / tot, corr / tot

    @torch.no_grad()
    def predict_proba(self, df, image_transform=None, data_root=".", bs=128):
        self.model.eval()
        ds = ManifestDS(df, data_root, tensor_tf(self.img), image_transform)
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=4)
        out = []
        for x, _ in dl:
            out.append(self.model(x.to(self.device)).softmax(1).cpu().numpy())
        return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", default="resnet50", choices=list(ARCHS))
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    run_dir = os.path.join(args.out, args.run_id); os.makedirs(run_dir, exist_ok=True)
    classes = load_classes(os.path.join(args.data, "classes_500.json"))
    names = [c["name"] for c in classes["classes"]]
    train_rows = read_manifest(os.path.join(args.data, "manifests", "train.csv"))
    val_rows = read_manifest(os.path.join(args.data, "manifests", "val.csv"))
    test_rows = read_manifest(os.path.join(args.data, "manifests", "test.csv"))

    m = DeepMethod(args.arch, args.pretrained)
    m.fit(train_rows, val_rows, args.data, epochs=args.epochs, lr=args.lr,
          bs=args.bs, augment=not args.no_aug, out=run_dir, resume=args.resume)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({"history": m.history, "train_seconds": m.train_seconds}, f, indent=2)

    t0 = time.time()
    probs = m.predict_proba(test_rows, data_root=args.data)
    test_seconds = time.time() - t0
    y = [r["class_id"] for r in test_rows]
    res = save_result(args.out, args.run_id, m.name, y, probs,
                      m.train_seconds, test_seconds, names)
    print(json.dumps({k: res[k] for k in ("top1", "top5", "macro_f1")}, indent=2),
          "->", run_dir)


if __name__ == "__main__":
    main()

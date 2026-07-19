"""Deep pipeline: train a CNN either from scratch or from ImageNet-pretrained weights.

Logs train/val loss & acc each epoch (for the required training curves) and saves
checkpoints (weights + optimizer) so training is resumable.

Example:
  python deep_cnn.py --data ../data/subset500 --arch resnet50 --pretrained \
      --epochs 30 --out ../results/resnet50_pretrained
  python deep_cnn.py --data ../data/subset500 --arch resnet50 \
      --epochs 60 --out ../results/resnet50_scratch     # from scratch
"""
import argparse, json, os, time
import torch, torch.nn as nn
from torchvision import models
from dataset import get_loaders

ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def build_model(arch, n_classes, pretrained):
    ctor, weights = ARCHS[arch]
    model = ctor(weights=weights if pretrained else None)
    if arch.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:  # efficientnet
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", default="resnet50", choices=list(ARCHS))
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--no_aug", action="store_true", help="ablation: turn off augmentation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders, classes = get_loaders(args.data, args.img, args.bs, augment=not args.no_aug)
    model = build_model(args.arch, len(classes), args.pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    start_ep, history, best = 0, [], 0.0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["opt"])
        start_ep, history, best = ck["epoch"] + 1, ck["history"], ck["best"]
        print(f"Resumed from epoch {start_ep}")

    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], device, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], device, criterion)
        scheduler.step()
        history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"ep{ep:02d} train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
        ck = {"model": model.state_dict(), "opt": optimizer.state_dict(),
              "epoch": ep, "history": history, "best": best, "classes": classes}
        torch.save(ck, os.path.join(args.out, "last.pt"))
        if va_acc > best:
            best = va_acc; torch.save(ck, os.path.join(args.out, "best.pt"))

    train_seconds = time.time() - t0
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"history": history, "train_seconds": train_seconds,
                   "arch": args.arch, "pretrained": args.pretrained,
                   "augment": not args.no_aug}, f, indent=2)
    print(f"Done. best val acc {best:.3f} in {train_seconds:.0f}s. -> {args.out}")
    print("Next: python evaluate.py --data", args.data, "--ckpt",
          os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()

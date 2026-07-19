"""[ADVANCED #2] Test-time degradation robustness — performance-vs-severity curves.

Model is FIXED (no retraining). Degrade ONLY the test images across several
severities per corruption type, and report how top-1 & macro-F1 fall.
This is the risk/robustness study — 4+ corruption types required.

  python robustness.py --data ../data/subset500 --ckpt ../results/resnet50_pretrained/best.pt
Outputs results/robustness.json (+ you plot curves from it).
"""
import argparse, io, json, os
import numpy as np, torch
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision import datasets, transforms
from deep_cnn import build_model
from dataset import IMAGENET_MEAN, IMAGENET_STD

# --- corruptions: each maps a PIL image + severity(0..1-ish) -> PIL image ---
def gaussian_noise(img, sev):
    a = np.asarray(img).astype(np.float32)
    a += np.random.normal(0, sev * 255, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def gaussian_blur(img, sev):
    return img.filter(ImageFilter.GaussianBlur(radius=sev))

def motion_blur(img, sev):
    # crude horizontal motion via repeated box blur
    k = max(1, int(sev))
    out = img
    for _ in range(k):
        out = out.filter(ImageFilter.BLUR)
    return out

def brightness(img, sev):
    a = np.asarray(img).astype(np.float32) * (1 - sev)  # darken
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def jpeg(img, sev):
    q = max(5, int(95 - sev * 90))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")

CORRUPTIONS = {
    "gaussian_noise": (gaussian_noise, [0.02, 0.05, 0.1, 0.2, 0.35]),
    "gaussian_blur":  (gaussian_blur,  [1, 2, 3, 5, 8]),
    "motion_blur":    (motion_blur,    [1, 2, 3, 4, 6]),
    "brightness":     (brightness,     [0.2, 0.4, 0.6, 0.75, 0.9]),
    "jpeg":           (jpeg,           [0.2, 0.4, 0.6, 0.8, 0.95]),
}


class CorruptTest(datasets.ImageFolder):
    def __init__(self, root, fn, sev, img=224):
        super().__init__(root)
        self.fn, self.sev = fn, sev
        self.norm = transforms.Compose([
            transforms.Resize(int(img * 1.14)), transforms.CenterCrop(img),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = self.loader(path).convert("RGB")
        if self.fn is not None:
            img = self.fn(img, self.sev)
        return self.norm(img), y


@torch.no_grad()
def eval_set(model, ds, device, n):
    dl = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=4)
    ys, ps = [], []
    for x, y in dl:
        ps.append(model(x.to(device)).argmax(1).cpu().numpy()); ys.append(y.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    _, _, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return accuracy_score(y, p), f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--out", default="../results/robustness.json")
    args = ap.parse_args()
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device); n = len(ck["classes"])
    model = build_model(args.arch, n, pretrained=False).to(device).eval()
    model.load_state_dict(ck["model"])
    test_root = f"{args.data}/test"

    results = {"clean": {}}
    a0, f0 = eval_set(model, CorruptTest(test_root, None, 0), device, n)
    results["clean"] = {"top1": a0, "macro_f1": f0}
    print(f"clean  top1={a0:.3f}  f1={f0:.3f}")
    for name, (fn, sevs) in CORRUPTIONS.items():
        results[name] = []
        for s in sevs:
            a, f = eval_set(model, CorruptTest(test_root, fn, s), device, n)
            results[name].append({"severity": s, "top1": a, "macro_f1": f})
            print(f"{name:14s} sev={s:<5} top1={a:.3f}  f1={f:.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved ->", args.out, "  (plot top1/f1 vs severity per corruption)")


if __name__ == "__main__":
    main()

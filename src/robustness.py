"""[E / ADVANCED #2] Test-time degradation robustness — via the Method contract.

E provides CORRUPTIONS (callable(PIL,sev)->PIL). Each method OWNER runs its own model:
    m.predict_proba(test_rows, image_transform=make(corruption, sev))
Model fixed, test-only, no retrain. Produces top-1 & macro-F1 vs severity per corruption.

  python robustness.py --data ../data --ckpt ../results/resnet50_pretrained_42/best.pt \
      --arch resnet50 --out ../results/robustness_resnet50.json
"""
import argparse, functools, io, json, os
import numpy as np
from PIL import ImageFilter
from common import read_manifest, load_classes, compute_metrics


def gaussian_noise(img, sev):
    a = np.asarray(img).astype(np.float32) + np.random.normal(0, sev * 255, (np.asarray(img)).shape)
    from PIL import Image
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def gaussian_blur(img, sev): return img.filter(ImageFilter.GaussianBlur(radius=sev))

def motion_blur(img, sev):
    out = img
    for _ in range(max(1, int(sev))):
        out = out.filter(ImageFilter.BLUR)
    return out

def brightness(img, sev):
    from PIL import Image
    a = np.asarray(img).astype(np.float32) * (1 - sev)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def jpeg(img, sev):
    from PIL import Image
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=max(5, int(95 - sev * 90))); buf.seek(0)
    return Image.open(buf).convert("RGB")

CORRUPTIONS = {
    "gaussian_noise": (gaussian_noise, [0.02, 0.05, 0.1, 0.2, 0.35]),
    "gaussian_blur":  (gaussian_blur,  [1, 2, 3, 5, 8]),
    "motion_blur":    (motion_blur,    [1, 2, 3, 4, 6]),
    "brightness":     (brightness,     [0.2, 0.4, 0.6, 0.75, 0.9]),
    "jpeg":           (jpeg,           [0.2, 0.4, 0.6, 0.8, 0.95]),
}


def make(fn, sev):
    return functools.partial(fn, sev=sev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--out", default="../results/robustness.json")
    args = ap.parse_args()
    np.random.seed(0)

    import torch
    from deep_cnn import DeepMethod
    m = DeepMethod(args.arch, pretrained=False)
    ck = torch.load(args.ckpt, map_location=m.device)
    m.model.load_state_dict(ck["model"])

    test_rows = read_manifest(os.path.join(args.data, "manifests", "test.csv"))
    y = [r["class_id"] for r in test_rows]
    names = [c["name"] for c in load_classes(os.path.join(args.data, "classes_500.json"))["classes"]]

    def evalp(itf):
        probs = m.predict_proba(test_rows, image_transform=itf, data_root=args.data)
        met, _ = compute_metrics(y, probs, names)
        return met["top1"], met["macro_f1"]

    results = {}
    a0, f0 = evalp(None)
    results["clean"] = {"top1": a0, "macro_f1": f0}
    print(f"clean top1={a0:.3f} f1={f0:.3f}")
    for name, (fn, sevs) in CORRUPTIONS.items():
        results[name] = []
        for s in sevs:
            a, f = evalp(make(fn, s))
            results[name].append({"severity": s, "top1": a, "macro_f1": f})
            print(f"{name:14s} sev={s:<5} top1={a:.3f} f1={f:.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print("Saved ->", args.out, " (plot top1/f1 vs severity per corruption)")


if __name__ == "__main__":
    main()

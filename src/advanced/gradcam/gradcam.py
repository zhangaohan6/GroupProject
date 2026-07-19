"""[E / ADVANCED #1] Grad-CAM — manifest + contract load_image based. Organism vs background,
correct vs wrong, confusable pairs. Every figure needs an interpretation in the report.

  python -m src.advanced.gradcam.gradcam --arch resnet50 \
      --ckpt data/checkpoints/transfer_resnet50__res50_ft/best.pt --n 24 --out figures/gradcam.png
"""
import argparse, os
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from src.common.io import load_image
from src.common.manifest import read_manifest, load_classes
from src.methods.deepnet import DeepMethod, _tf, MEAN, STD


class GradCAM:
    def __init__(self, model, layer):
        self.model = model.eval(); self.a = self.g = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "a", o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "g", go[0].detach()))

    def __call__(self, x):
        out = self.model(x); cls = out.argmax(1)
        self.model.zero_grad(); out.gather(1, cls.view(-1, 1)).sum().backward()
        cam = F.relu((self.g.mean((2, 3), keepdim=True) * self.a).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        cam = (cam - cam.amin((1, 2), keepdim=True)) / (cam.amax((1, 2), keepdim=True) + 1e-6)
        return cam.cpu().numpy(), cls.cpu().numpy()


def denorm(t):
    m = torch.tensor(MEAN).view(3, 1, 1); s = torch.tensor(STD).view(3, 1, 1)
    return (t * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="figures/gradcam.png")
    a = ap.parse_args()
    import matplotlib.pyplot as plt
    m = DeepMethod(a.arch, pretrained=False, data_root=a.data); m.load(a.ckpt)
    target = m.model.layer4[-1] if a.arch.startswith("resnet") else m.model.features[-1]
    cam = GradCAM(m.model, target)
    names = [c["species"] for c in load_classes(f"{a.data}/classes_500.json")["classes"]]
    rows = read_manifest(f"{a.data}/manifests/test.csv")
    idx = np.random.RandomState(0).choice(len(rows), a.n, replace=False)
    tf = _tf(224)
    cols = 4; r = (a.n + cols - 1) // cols
    plt.figure(figsize=(cols * 3, r * 3))
    for k, i in enumerate(idx):
        row = rows.iloc[i]
        arr = load_image(os.path.join(a.data, "images_256", row["filepath"]))
        x = tf(Image.fromarray(arr)).unsqueeze(0).to(m.device)
        heat, pred = cam(x)
        plt.subplot(r, cols, k + 1); plt.imshow(denorm(x[0].cpu())); plt.imshow(heat[0], cmap="jet", alpha=0.45)
        ok = "OK" if pred[0] == row["class_id"] else "WRONG"
        plt.title(f"gt:{names[row['class_id']][:12]} pr:{names[pred[0]][:12]} {ok}", fontsize=6); plt.axis("off")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.tight_layout(); plt.savefig(a.out, dpi=150)
    print("Saved", a.out, "— interpret: heatmap on organism or background?")


if __name__ == "__main__":
    main()

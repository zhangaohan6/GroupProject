"""[E / ADVANCED #1] Grad-CAM — manifest + common.load_image based.

Overlays class-activation heatmaps: organism vs background, correct vs wrong,
confusable species pairs. Every figure must get an interpretation in the report.

  python gradcam.py --data ../data --ckpt ../results/resnet50_pretrained_42/best.pt \
      --arch resnet50 --n 24 --out ../results/gradcam
"""
import argparse, os
import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from common import read_manifest, load_classes, load_image
from dataset import tensor_tf, IMAGENET_MEAN, IMAGENET_STD
from deep_cnn import DeepMethod


class GradCAM:
    def __init__(self, model, layer):
        self.model = model.eval(); self.a = None; self.g = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "a", o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "g", go[0].detach()))

    def __call__(self, x, cls=None):
        out = self.model(x)
        cls = out.argmax(1) if cls is None else cls
        self.model.zero_grad(); out.gather(1, cls.view(-1, 1)).sum().backward()
        w = self.g.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.a).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        cam = (cam - cam.amin((1, 2), keepdim=True)) / (cam.amax((1, 2), keepdim=True) + 1e-6)
        return cam.cpu().numpy(), cls.cpu().numpy()


def denorm(t):
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1); s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="../results/gradcam")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    m = DeepMethod(args.arch, pretrained=False)
    ck = torch.load(args.ckpt, map_location=m.device); m.model.load_state_dict(ck["model"])
    target = m.model.layer4[-1] if args.arch.startswith("resnet") else m.model.features[-1]
    cam = GradCAM(m.model, target)

    names = [c["name"] for c in load_classes(os.path.join(args.data, "classes_500.json"))["classes"]]
    rows = read_manifest(os.path.join(args.data, "manifests", "test.csv"))
    idx = np.random.RandomState(0).choice(len(rows), args.n, replace=False)
    tf = tensor_tf(224)

    cols = 4; r = (args.n + cols - 1) // cols
    plt.figure(figsize=(cols * 3, r * 3))
    for k, i in enumerate(idx):
        row = rows[i]
        x = tf(load_image(os.path.join(args.data, row["path"]))).unsqueeze(0).to(m.device)
        heat, pred = cam(x)
        plt.subplot(r, cols, k + 1)
        plt.imshow(denorm(x[0].cpu())); plt.imshow(heat[0], cmap="jet", alpha=0.45)
        ok = "OK" if pred[0] == row["class_id"] else "WRONG"
        plt.title(f"gt:{names[row['class_id']][:12]} pr:{names[pred[0]][:12]} {ok}", fontsize=6)
        plt.axis("off")
    plt.tight_layout(); plt.savefig(f"{args.out}/gradcam_grid.png", dpi=150)
    print("Saved ->", f"{args.out}/gradcam_grid.png",
          "\nInterpret: does the heatmap land on the organism or the background?")


if __name__ == "__main__":
    main()

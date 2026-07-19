"""[ADVANCED #1] Grad-CAM: which regions drive the CNN's prediction?

Overlays class-activation heatmaps to show whether the model attends to the
organism vs background. Compare correct vs wrong, and confusable species pairs.

  python gradcam.py --data ../data/subset500 --ckpt ../results/resnet50_pretrained/best.pt \
      --n 24 --out ../results/gradcam
"""
import argparse, os
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from deep_cnn import build_model
from dataset import build_transforms, IMAGENET_MEAN, IMAGENET_STD
from torchvision import datasets


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval(); self.acts = None; self.grads = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grads", go[0].detach()))

    def __call__(self, x, cls=None):
        out = self.model(x)
        cls = out.argmax(1) if cls is None else cls
        self.model.zero_grad()
        out.gather(1, cls.view(-1, 1)).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)         # GAP over gradients
        cam = F.relu((w * self.acts).sum(1))                  # weighted sum of feature maps
        cam = F.interpolate(cam.unsqueeze(1), x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam = (cam - cam.amin((1, 2), keepdim=True)) / (cam.amax((1, 2), keepdim=True) + 1e-6)
        return cam.cpu().numpy(), cls.cpu().numpy(), out.softmax(1).detach().cpu().numpy()


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device); classes = ck["classes"]
    model = build_model(args.arch, len(classes), pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    target = model.layer4[-1] if args.arch.startswith("resnet") else model.features[-1]
    cam_fn = GradCAM(model, target)

    tf = build_transforms(train=False)
    ds = datasets.ImageFolder(f"{args.data}/test", transform=tf)
    idxs = np.random.RandomState(0).choice(len(ds), args.n, replace=False)

    cols = 4; rows = (args.n + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for k, i in enumerate(idxs):
        x, y = ds[i]; x = x.unsqueeze(0).to(device)
        cam, pred, prob = cam_fn(x)
        img = denorm(x[0].cpu())
        plt.subplot(rows, cols, k + 1)
        plt.imshow(img); plt.imshow(cam[0], cmap="jet", alpha=0.45)
        ok = "OK" if pred[0] == y else "WRONG"
        plt.title(f"gt{classes[y]} pr{classes[pred[0]]} {ok}", fontsize=7)
        plt.axis("off")
    plt.tight_layout(); plt.savefig(f"{args.out}/gradcam_grid.png", dpi=150)
    print("Saved ->", f"{args.out}/gradcam_grid.png",
          "\nDiscuss: does the heatmap land on the organism or the background?")


if __name__ == "__main__":
    main()

"""[C/D shared] Deep CNN implementing Method. scratch/run.py and transfer/run.py are thin
wrappers over this (pretrained flag differs). Honors the ndarray image_transform hook."""
import os, time
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from src.methods.base import Method, N_CLASSES
from src.common.io import load_image

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def _tf(img=224, train=False, augment=True):
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose([transforms.Resize(int(img * 1.14)), transforms.CenterCrop(img),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def _build(arch, n, pretrained):
    ctor, w = ARCHS[arch]
    m = ctor(weights=w if pretrained else None)
    if arch.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, n)
    else:
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    return m


class _DS(torch.utils.data.Dataset):
    def __init__(self, df, data_root, tf, image_transform):
        self.df, self.root, self.tf, self.itf = df.reset_index(drop=True), data_root, tf, image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        arr = load_image(os.path.join(self.root, "images_256", r["filepath"]))  # HWC uint8 RGB
        if self.itf is not None:
            arr = self.itf(arr)                                                  # contract §3.2
        return self.tf(Image.fromarray(arr)), int(r["class_id"])


class DeepMethod(Method):
    def __init__(self, arch="resnet50", pretrained=True, data_root="data", img=224, tag=None):
        self.arch, self.pretrained, self.root, self.img = arch, pretrained, data_root, img
        self.name = f"{'transfer' if pretrained else 'scratch'}_{arch}"
        self.tag = tag or self.name
        self.config = {"backbone": arch, "pretrained": pretrained, "img": img}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _build(arch, N_CLASSES, pretrained).to(self.device)
        self.history = []

    def fit(self, train_df, val_df, epochs=30, lr=1e-3, bs=64, augment=True, ckpt_dir=None, resume=""):
        self.config.update({"epochs": epochs, "lr": lr, "bs": bs, "augment": augment})
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        tr = torch.utils.data.DataLoader(_DS(train_df, self.root, _tf(self.img, True, augment), None),
                                         batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        va = torch.utils.data.DataLoader(_DS(val_df, self.root, _tf(self.img), None),
                                         batch_size=bs, num_workers=4, pin_memory=True)
        start, best, t0 = 0, 0.0, time.time()
        if resume and os.path.exists(resume):
            c = torch.load(resume, map_location=self.device)
            self.model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
            start, self.history, best = c["epoch"] + 1, c["history"], c["best"]
        for ep in range(start, epochs):
            trl, tra = self._epoch(tr, crit, opt); val, vaa = self._epoch(va, crit); sch.step()
            self.history.append({"epoch": ep, "train_loss": trl, "train_acc": tra,
                                 "val_loss": val, "val_acc": vaa})
            print(f"ep{ep:02d} train {trl:.3f}/{tra:.3f} val {val:.3f}/{vaa:.3f}")
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
                c = {"model": self.model.state_dict(), "opt": opt.state_dict(),
                     "epoch": ep, "history": self.history, "best": best, "arch": self.arch}
                torch.save(c, os.path.join(ckpt_dir, "last.pt"))
                if vaa > best:
                    best = vaa; torch.save(c, os.path.join(ckpt_dir, "best.pt"))
        self.train_seconds = time.time() - t0
        return self

    def _epoch(self, loader, crit, opt=None):
        train = opt is not None; self.model.train(train)
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
    def predict_proba(self, df, image_transform=None, bs=128):
        self.model.eval()
        dl = torch.utils.data.DataLoader(_DS(df, self.root, _tf(self.img), image_transform),
                                         batch_size=bs, num_workers=4)
        return np.concatenate([self.model(x.to(self.device)).softmax(1).cpu().numpy() for x, _ in dl])

    def save(self, path):
        torch.save({"model": self.model.state_dict(), "arch": self.arch,
                    "config": self.config}, path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device)["model"]); return self

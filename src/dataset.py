"""Manifest-driven dataset (CONTRACTS §2,§3). Supports the image_transform hook so
E's robustness sweep runs through the same path on every method."""
import os
import torch
from torchvision import transforms
from common import load_image, read_manifest

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def tensor_tf(img_size=224, train=False, augment=True):
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)), transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ManifestDS(torch.utils.data.Dataset):
    """rows: list of {path,class_id}. image_transform: callable(PIL)->PIL applied
    after load_image, before the tensor transform (this is the robustness hook)."""
    def __init__(self, rows, data_root, tf, image_transform=None):
        self.rows, self.root, self.tf, self.itf = rows, data_root, tf, image_transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = load_image(os.path.join(self.root, r["path"]))
        if self.itf is not None:
            img = self.itf(img)
        return self.tf(img), r["class_id"]


def loader_from_manifest(data_root, split, img_size=224, batch_size=64, num_workers=4,
                         train=False, augment=True, image_transform=None):
    rows = read_manifest(os.path.join(data_root, "manifests", f"{split}.csv"))
    ds = ManifestDS(rows, data_root, tensor_tf(img_size, train, augment), image_transform)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train,
                                       num_workers=num_workers, pin_memory=True), rows

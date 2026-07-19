"""ImageFolder-style loaders for the built subset (train/val/test dirs of class folders)."""
import torch
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size=224, train=True, augment=True):
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_loaders(data_root, img_size=224, batch_size=64, num_workers=4, augment=True):
    """data_root/{train,val,test}/<class>/<img>. Returns (loaders, class_names)."""
    tf_train = build_transforms(img_size, train=True, augment=augment)
    tf_eval = build_transforms(img_size, train=False)
    ds = {s: datasets.ImageFolder(f"{data_root}/{s}",
                                  transform=tf_train if s == "train" else tf_eval)
          for s in ["train", "val", "test"]}
    loaders = {s: torch.utils.data.DataLoader(
        ds[s], batch_size=batch_size, shuffle=(s == "train"),
        num_workers=num_workers, pin_memory=True) for s in ds}
    return loaders, ds["train"].classes

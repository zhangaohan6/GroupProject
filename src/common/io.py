"""[A] Unified IO — CONTRACTS §3.3, §7. Everyone imports load_image/set_seed from here."""
import os
import random
import numpy as np
from PIL import Image

SEED = 42
N_CLASSES = 500


def load_image(path):
    """THE image entry point. Returns HWC / uint8 / RGB ndarray (CONTRACTS §3.3).
    Never use cv2.imread / Image.open anywhere else."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def set_seed(seed=SEED):
    """Seed random / numpy / torch(+cuda) — call at the top of every train/sample script."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

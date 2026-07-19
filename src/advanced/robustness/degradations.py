"""[E] Degradation library — CONTRACTS §3.2 (operate on HWC uint8 RGB ndarray, online).
Each returns callable(ndarray)->ndarray usable as image_transform. 5 types x 5 severities.
"""
import io
import numpy as np
from PIL import Image, ImageFilter


def _pil(a): return Image.fromarray(a)
def _np(im): return np.asarray(im.convert("RGB"), np.uint8)


def gaussian_noise(sev):
    def f(a):
        return np.clip(a.astype(np.float32) + np.random.normal(0, sev * 255, a.shape), 0, 255).astype(np.uint8)
    return f

def gaussian_blur(sev):
    return lambda a: _np(_pil(a).filter(ImageFilter.GaussianBlur(radius=sev)))

def motion_blur(sev):
    def f(a):
        im = _pil(a)
        for _ in range(max(1, int(sev))):
            im = im.filter(ImageFilter.BLUR)
        return _np(im)
    return f

def brightness(sev):  # darken
    return lambda a: np.clip(a.astype(np.float32) * (1 - sev), 0, 255).astype(np.uint8)

def jpeg(sev):
    def f(a):
        buf = io.BytesIO(); _pil(a).save(buf, "JPEG", quality=max(5, int(95 - sev * 90))); buf.seek(0)
        return _np(Image.open(buf))
    return f

CORRUPTIONS = {
    "gaussian_noise": (gaussian_noise, [0.02, 0.05, 0.1, 0.2, 0.35]),
    "gaussian_blur":  (gaussian_blur,  [1, 2, 3, 5, 8]),
    "motion_blur":    (motion_blur,    [1, 2, 3, 4, 6]),
    "brightness":     (brightness,     [0.2, 0.4, 0.6, 0.75, 0.9]),
    "jpeg":           (jpeg,           [0.2, 0.4, 0.6, 0.8, 0.95]),
}

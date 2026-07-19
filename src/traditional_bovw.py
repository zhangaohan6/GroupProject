"""[B / ADVANCED-ready] Traditional pipeline as common.Method: BoVW(SIFT)+LinearSVC.

Implements predict_proba(df, image_transform) so it plugs into evaluate + robustness
exactly like the deep methods. Expected top-1 is low (3-10%) on 500 classes — that is a
valid finding (handcrafted features plateau on fine-grained), not a failure.

  python traditional_bovw.py --data ../data --k 512 --run_id bovw_sift512_42 --out ../results
Needs opencv-python.
"""
import argparse, json, os, time
import numpy as np, cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from common import Method, N_CLASSES, read_manifest, load_classes, save_result, load_image


def to_gray(img_pil, image_transform, size=256):
    if image_transform is not None:
        img_pil = image_transform(img_pil)
    g = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (size, size))


class BoVWMethod(Method):
    def __init__(self, data_root, k=512):
        self.name = f"bovw_sift{k}"
        self.root, self.k = data_root, k
        self.sift = cv2.SIFT_create()

    def _desc(self, path, image_transform=None):
        g = to_gray(load_image(os.path.join(self.root, path)), image_transform)
        _, d = self.sift.detectAndCompute(g, None)
        return d

    def _hist(self, path, image_transform=None):
        d = self._desc(path, image_transform)
        h = np.zeros(self.k, np.float32)
        if d is not None:
            for w in self.km.predict(d):
                h[w] += 1
            h /= (h.sum() + 1e-6)
        return h

    def fit(self, train_rows, val_rows=None, max_vocab_imgs=2000):
        rng = np.random.RandomState(0)
        sub = [train_rows[i] for i in rng.choice(len(train_rows),
               min(max_vocab_imgs, len(train_rows)), replace=False)]
        alld = [d for r in sub if (d := self._desc(r["path"])) is not None]
        self.km = MiniBatchKMeans(self.k, random_state=0, batch_size=10000, n_init=3)
        self.km.fit(np.vstack(alld))
        X = np.stack([self._hist(r["path"]) for r in train_rows])
        y = np.array([r["class_id"] for r in train_rows])
        self.scaler = StandardScaler().fit(X)
        self.clf = LinearSVC(C=1.0, max_iter=5000).fit(self.scaler.transform(X), y)
        return self

    def predict_proba(self, df, image_transform=None):
        X = np.stack([self._hist(r["path"], image_transform) for r in df])
        dec = self.clf.decision_function(self.scaler.transform(X))  # (N, seen_classes)
        # map to full 500-col space (classes seen by clf may be < 500)
        full = np.full((len(df), N_CLASSES), -1e9, np.float32)
        full[:, self.clf.classes_] = dec
        return softmax(full, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()
    names = [c["name"] for c in load_classes(os.path.join(args.data, "classes_500.json"))["classes"]]
    train = read_manifest(os.path.join(args.data, "manifests", "train.csv"))
    test = read_manifest(os.path.join(args.data, "manifests", "test.csv"))

    m = BoVWMethod(args.data, args.k)
    t0 = time.time(); m.fit(train); train_s = time.time() - t0
    t1 = time.time(); probs = m.predict_proba(test); test_s = time.time() - t1
    y = [r["class_id"] for r in test]
    res = save_result(args.out, args.run_id, m.name, y, probs, train_s, test_s, names)
    print(json.dumps({k: res[k] for k in ("top1", "top5", "macro_f1")}, indent=2))


if __name__ == "__main__":
    main()

"""[B] Traditional pipeline as Method: BoVW(SIFT)+LinearSVC. decision_function->softmax
for probs (CONTRACTS §3.4). Honors the ndarray image_transform hook.

  python -m src.methods.traditional.bovw --k 512 --tag sift512
Low top-1 (3-10%) is a valid finding, not a failure.
"""
import argparse, os, pickle, time
import numpy as np, cv2
from scipy.special import softmax
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from src.methods.base import Method, N_CLASSES, save_run
from src.common.io import load_image, set_seed
from src.common.manifest import read_manifest


class BoVWMethod(Method):
    def __init__(self, data_root="data", k=512, size=256):
        self.name, self.root, self.k, self.size = "traditional_bovw", data_root, k, size
        self.tag = f"sift{k}"
        self.config = {"features": "SIFT", "codebook": k, "classifier": "LinearSVC",
                       "proba": "softmax(decision_function)"}
        self.sift = cv2.SIFT_create()

    def _gray(self, filepath, itf):
        arr = load_image(os.path.join(self.root, "images_256", filepath))  # HWC uint8 RGB
        if itf is not None:
            arr = itf(arr)
        g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return cv2.resize(g, (self.size, self.size))

    def _desc(self, filepath, itf=None):
        _, d = self.sift.detectAndCompute(self._gray(filepath, itf), None)
        return d

    def _hist(self, filepath, itf=None):
        d = self._desc(filepath, itf); h = np.zeros(self.k, np.float32)
        if d is not None:
            for w in self.km.predict(d):
                h[w] += 1
            h /= (h.sum() + 1e-6)
        return h

    def fit(self, train_df, val_df=None, max_vocab=2000):
        rng = np.random.RandomState(0)
        idx = rng.choice(len(train_df), min(max_vocab, len(train_df)), replace=False)
        alld = [d for fp in train_df.iloc[idx]["filepath"] if (d := self._desc(fp)) is not None]
        self.km = MiniBatchKMeans(self.k, random_state=0, batch_size=10000, n_init=3).fit(np.vstack(alld))
        X = np.stack([self._hist(fp) for fp in train_df["filepath"]])
        y = train_df["class_id"].to_numpy()
        self.scaler = StandardScaler().fit(X)
        self.clf = LinearSVC(C=1.0, max_iter=5000).fit(self.scaler.transform(X), y)
        return self

    def predict_proba(self, df, image_transform=None):
        X = np.stack([self._hist(fp, image_transform) for fp in df["filepath"]])
        dec = self.clf.decision_function(self.scaler.transform(X))
        full = np.full((len(df), N_CLASSES), -1e9, np.float32)
        full[:, self.clf.classes_] = dec
        return softmax(full, axis=1)

    def save(self, path):
        pickle.dump({"km": self.km, "scaler": self.scaler, "clf": self.clf, "k": self.k}, open(path, "wb"))

    def load(self, path):
        s = pickle.load(open(path, "rb")); self.km, self.scaler, self.clf, self.k = (
            s["km"], s["scaler"], s["clf"], s["k"]); return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--tag", default="sift512")
    a = ap.parse_args(); set_seed()
    tr = read_manifest(f"{a.data}/manifests/train.csv"); te = read_manifest(f"{a.data}/manifests/test.csv")
    m = BoVWMethod(a.data, a.k)
    t0 = time.time(); m.fit(tr); ts = time.time() - t0
    t1 = time.time(); probs = m.predict_proba(te); ps = time.time() - t1
    res = save_run(".", m.name, a.tag, "test", te["class_id"].to_numpy(), probs, m.config, ts, ps)
    print("saved", res["run"], {k: res["metrics"][k] for k in ("top1", "top5", "macro_f1")})


if __name__ == "__main__":
    main()

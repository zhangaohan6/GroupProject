"""Traditional pipeline: Bag-of-Visual-Words (SIFT) + SVM.

A genuinely different (non-deep) baseline. Also supports HOG/LBP by swapping
the descriptor (see extract_features). Reports top-1 + macro-F1 + timing.

  python traditional_bovw.py --data ../data/subset500 --k 512 --out ../results/bovw
Needs: opencv-python, scikit-learn, scikit-image
"""
import argparse, glob, json, os, time
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def list_split(root, split):
    items = []
    for cls in sorted(os.listdir(f"{root}/{split}")):
        for p in glob.glob(f"{root}/{split}/{cls}/*"):
            items.append((p, cls))
    return items


def sift_descriptors(path, sift, size=256):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size, size))
    _, des = sift.detectAndCompute(img, None)
    return des  # (n_kp, 128) or None


def build_vocab(train, sift, k, max_imgs=2000):
    rng = np.random.RandomState(0)
    sample = [train[i] for i in rng.choice(len(train), min(max_imgs, len(train)), replace=False)]
    alld = [d for p, _ in sample if (d := sift_descriptors(p, sift)) is not None]
    alld = np.vstack(alld)
    print(f"KMeans on {alld.shape[0]} descriptors -> {k} words")
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=10000, n_init=3)
    km.fit(alld)
    return km


def bovw_hist(path, sift, km, k):
    des = sift_descriptors(path, sift)
    h = np.zeros(k, np.float32)
    if des is not None:
        for w in km.predict(des):
            h[w] += 1
        h /= (h.sum() + 1e-6)
    return h


def featurize(items, sift, km, k):
    return np.stack([bovw_hist(p, sift, km, k) for p, _ in items]), \
           np.array([c for _, c in items])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=512, help="visual-word vocab size")
    ap.add_argument("--out", default="../results/bovw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sift = cv2.SIFT_create()

    train = list_split(args.data, "train")
    test = list_split(args.data, "test")
    t0 = time.time()
    km = build_vocab(train, sift, args.k)
    Xtr, ytr = featurize(train, sift, km, args.k)
    scaler = StandardScaler().fit(Xtr)
    clf = LinearSVC(C=1.0, max_iter=5000).fit(scaler.transform(Xtr), ytr)
    train_secs = time.time() - t0

    t1 = time.time()
    Xte, yte = featurize(test, sift, km, args.k)
    pred = clf.predict(scaler.transform(Xte))
    test_secs = time.time() - t1

    top1 = accuracy_score(yte, pred)
    _, _, f1, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    res = {"method": f"BoVW-SIFT-{args.k}+LinearSVC", "top1": top1, "macro_f1": f1,
           "train_seconds": train_secs, "test_seconds": test_secs}
    print(json.dumps(res, indent=2))
    with open(f"{args.out}/eval.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

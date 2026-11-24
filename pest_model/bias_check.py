"""촬영 거리·구도 편향 점검.

병반이 보이지 않을 만큼 줄인 16x16 사진과 색 히스토그램만으로 로지스틱 회귀를 학습한다.
이것으로도 test 성능이 높으면 CNN 점수의 상당 부분은 병반이 아니라
'가까이 찍었나/배경이 무엇인가'에서 나온 것이라고 봐야 한다.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["healthy", "shot_hole", "brown_rot"]


def feats(path, kind):
    im = Image.open(path).convert("RGB")
    if kind == "tiny16":
        return np.asarray(im.resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel() / 255
    hsv = np.asarray(im.convert("HSV").resize((64, 64)), dtype=np.int32)
    hs = [np.bincount(hsv[..., c].ravel() * 16 // 256, minlength=16) for c in range(3)]
    return np.concatenate(hs).astype(np.float32) / (64 * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(Path(args.data) / "splits.csv")))
    res = {}
    for kind in ["tiny16", "hsv_hist"]:
        X = {s: [] for s in ("train", "val", "test")}
        Y = {s: [] for s in X}
        for r in rows:
            X[r["split"]].append(feats(Path(args.data) / "images" / r["file"], kind))
            Y[r["split"]].append(CLASSES.index(r["cls"]))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"))
        clf.fit(np.array(X["train"]), Y["train"])
        p = clf.predict(np.array(X["test"]))
        res[kind] = dict(test_acc=round(accuracy_score(Y["test"], p), 4),
                         test_macro_f1=round(f1_score(Y["test"], p, average="macro"), 4))
        print(kind, res[kind])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "bias_check.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

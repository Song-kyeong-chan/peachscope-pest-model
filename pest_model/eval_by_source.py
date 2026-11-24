"""출처 지름길 점검 (실험 2-B).

B 데이터는 세균구멍병이 zeerox에서만, 잿빛무늬병이 주로 Islamia에서 나온다.
모델이 병반 대신 '어느 데이터에서 온 사진인가'를 배웠다면,
같은 정상 과실이라도 Islamia 쪽 정상을 잿빛무늬병으로, zeerox 쪽 잿빛무늬병을 세균구멍병으로 더 자주 틀릴 것이다.

1) CNN 예측을 (출처, 정답 클래스)별로 나눠 재현율과 오답 방향을 본다.
2) 저해상도(16×16) 입력으로 '출처'를 맞히는 로지스틱 회귀 → 출처를 얼마나 쉽게 구별할 수 있는지(지름길의 재료) 본다.
결과: <out>/source_check_<arch>.json
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms as T

CLASSES = ["healthy", "shot_hole", "brown_rot"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="clean_b_crops")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D = Path(args.data)
    rows = list(csv.DictReader(open(D / "crops.csv")))
    test = [r for r in rows if r["split"] == "test"]

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = timm.create_model(args.arch, pretrained=False, num_classes=3)
    m.load_state_dict(torch.load(args.weights, map_location="cpu"))
    m.eval().to(dev)
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    preds = []
    with torch.no_grad():
        for i in range(0, len(test), 64):
            x = torch.stack([tf(Image.open(D / r["path"]).convert("RGB")) for r in test[i:i + 64]]).to(dev)
            preds += m(x).argmax(1).cpu().tolist()

    by = defaultdict(Counter)
    for r, p in zip(test, preds):
        by[(r["source"], r["label"])][CLASSES[p]] += 1
    cnn = {}
    print("출처별 test 예측 (정답 → 예측 분포)")
    for (src, lab), c in sorted(by.items()):
        n = sum(c.values())
        cnn[f"{src}/{lab}"] = {"n": n, "recall": round(c[lab] / n, 4), "pred": dict(c)}
        print(f"  {src:8s} {lab:10s} n={n:4d} 재현율 {c[lab] / n:.3f}  예측 {dict(c)}")

    # 저해상도로 출처 맞히기 (같은 클래스 안에서: 정상, 잿빛무늬병은 두 출처에 다 있다)
    def tiny(r):
        return np.asarray(Image.open(D / r["path"]).convert("RGB").resize((16, 16), Image.BILINEAR), np.float32).ravel() / 255
    src_res = {}
    for lab in ("healthy", "brown_rot"):
        tr = [r for r in rows if r["split"] == "train" and r["label"] == lab]
        te = [r for r in rows if r["split"] == "test" and r["label"] == lab]
        if len({r["source"] for r in tr}) < 2 or len({r["source"] for r in te}) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"))
        clf.fit(np.array([tiny(r) for r in tr]), [r["source"] for r in tr])
        acc = accuracy_score([r["source"] for r in te], clf.predict(np.array([tiny(r) for r in te])))
        base = max(Counter(r["source"] for r in te).values()) / len(te)
        src_res[lab] = {"tiny16_source_acc": round(acc, 4), "majority_baseline": round(base, 4), "n_test": len(te)}
        print(f"  [{lab}] 16×16로 출처 맞히기 정확도 {acc:.3f} (다수 클래스 기준선 {base:.3f}, n={len(te)})")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, f"source_check_{args.arch}.json").write_text(
        json.dumps({"cnn_by_source": cnn, "source_from_tiny16": src_res}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""test2 정답 v2: others 149개를 review 폴더 판정으로 바꿔 기존 예측(preds.csv)을 다시 채점한다.

- ~/peachscope-data/test2_others/review/<폴더>/<idx>_<크롭 stem>__D-*.jpg 의 폴더가 새 정답이다.
- _애매·_제외는 평가에서 뺀다(사용자 결정). others가 아닌 박스(정상·잿빛무늬병·반점·탄저병)는 그대로 둔다.
- 재학습 없이 각 모델 결과 폴더의 preds.csv(split=test2)를 쓴다.
출력: results/test2_others/labels_v2.csv, results/test2_others/rescore.json
"""
import argparse
import collections
import csv
import json
from math import comb
from pathlib import Path

REVIEW = Path.home() / "peachscope-data/test2_others/review"
CLS = ["healthy", "spot", "rot", "anthracnose", "damage"]


def new_labels():
    out = {}
    for d in REVIEW.iterdir():
        if d.is_dir():
            for f in d.glob("*.jpg"):
                stem = f.name.split("_", 1)[1].split("__D-")[0]
                out[f"test2/{stem}.jpg"] = d.name if d.name in CLS else None
    return out


def mcnemar(a, b):
    n = a + b; k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="results/ 아래 결과 폴더 이름 (첫 번째가 McNemar 기준)")
    a = ap.parse_args()
    nl = new_labels()
    base = [r for r in csv.DictReader(open("results/exp7/labels.csv")) if r["split"] == "test2"]
    labels = {}
    for r in base:
        lab = nl.get(r["file"], "missing") if r["orig"] == "others" else r["cls"]
        assert lab != "missing", r["file"]
        labels[r["file"]] = lab
    Path("results/test2_others").mkdir(exist_ok=True)
    with open("results/test2_others/labels_v2.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "orig", "cls_v1", "cls_v2"])
        for r in base:
            w.writerow([r["file"], r["orig"], r["cls"], labels[r["file"]] or "excluded"])
    keep = {k: v for k, v in labels.items() if v}
    print("test2 v2:", len(keep), "개 (제외", len(labels) - len(keep), ")", dict(collections.Counter(keep.values())))
    res, ok = {}, {}
    for run in a.runs:
        P = {r["file"]: r["pred"] for r in csv.DictReader(open(f"results/{run}/preds.csv")) if r["split"] == "test2"}
        ok[run] = {f: P[f] == y for f, y in keep.items()}
        per = {c: f"{sum(ok[run][f] for f, y in keep.items() if y == c)}/{sum(y == c for y in keep.values())}" for c in CLS}
        abn = [f for f, y in keep.items() if y != "healthy"]
        res[run] = dict(correct=sum(ok[run].values()), n=len(keep), per_class=per,
                        anomaly=f"{sum(P[f] != 'healthy' for f in abn)}/{len(abn)}",
                        healthy_kept=f"{sum(P[f] == 'healthy' for f, y in keep.items() if y == 'healthy')}/{sum(y == 'healthy' for y in keep.values())}",
                        others_only=f"{sum(ok[run][f] for f in keep if f in nl)}/{sum(f in nl for f in keep)}")
    b0 = a.runs[0]
    for run in a.runs[1:]:
        x = sum(ok[b0][f] and not ok[run][f] for f in keep); y = sum(ok[run][f] and not ok[b0][f] for f in keep)
        res[run]["mcnemar_vs_" + b0] = dict(only_base=x, only_this=y, p=round(mcnemar(x, y), 4))
    for run, r in res.items():
        print(run, json.dumps(r, ensure_ascii=False))
    (Path("results/test2_others") / "rescore.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""실험 4 라벨: 반점병 통합 + 탄저병 분리 (label_scheme.py), 팀 검수 전 '얼추 정리' 버전.

- 이미지는 exp3_crops/images를 그대로 쓰고 라벨 csv만 새로 만든다.
- train/val: final_labels.csv에서 제외(exclude)만 빼고, 확정(ok)은 확정 라벨, 검수 대상(review)은 rough_label(모델이 0.7 이상 확신하면 모델 판정, 아니면 제안 라벨)을 쓴다.
  (확정 라벨만 쓰면 경계 사례가 빠져 NCPMS 성능이 떨어졌다: 실험 기록 '입력 크기 비교' 참고)
- test: NCPMS 91장을 실험 4 클래스로 다시 매핑한다(원래 병명은 orig 열).
출력: results/exp4/labels.csv (file, split, cls, status, orig)
"""
import collections
import csv
import json
from pathlib import Path

from label_scheme import NCPMS4, ant_photos, rough_label, to4

H = Path.home() / "peachscope-data"


def main():
    ant = ant_photos()
    rows = []
    for r in csv.DictReader(open("results/relabel_trainval/final_labels.csv")):
        if r["status"] == "exclude":
            continue
        cls = rough_label(r, ant) if r["status"] == "review" else to4(r["final"], r["file"], ant)
        rows.append(dict(file=r["file"], split=r["split"], cls=cls, status=r["status"], orig=r["final"]))
    orig = {Path(e["file"]).stem: e["cls"] for e in json.load(open(H / "ncpms/fruit_images.json"))}
    for r in csv.DictReader(open(H / "exp3_crops/splits.csv")):
        if r["split"] == "test":
            o = orig[Path(r["file"]).stem[3:]]
            rows.append(dict(file=r["file"], split="test", cls=NCPMS4.get(o, "others"), status="ncpms", orig=o))
    out = Path("results/exp4"); out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    c = collections.Counter((r["split"], r["cls"]) for r in rows)
    for s in ("train", "val", "test"):
        print(s, {k: v for (ss, k), v in sorted(c.items()) if ss == s})
    print("review(제안 라벨) 포함", sum(r["status"] == "review" for r in rows))


if __name__ == "__main__":
    main()

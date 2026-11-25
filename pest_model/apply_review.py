"""팀 검수 결과(review 폴더)를 읽어 라벨에 반영한다.

- 파일 이름 앞부분(`_모델-` 앞)이 크롭 id(rid), 들어 있는 폴더가 검수 라벨이다. `_제외`는 학습에서 뺀다.
- 검수 전 폴더(rough_label)와 비교해 이동 표를 만든다.
출력: results/relabel_trainval/reviewed_labels.csv (file, rid, before, after)
"""
import collections
import csv
from pathlib import Path

from label_scheme import ant_photos, rough_label

REVIEW = Path.home() / "peachscope-data/relabel_trainval/review"


def main():
    rows = {r["rid"]: r for r in csv.DictReader(open("results/relabel_trainval/final_labels.csv")) if r["status"] == "review"}
    ant = ant_photos()
    out, seen = [], set()
    for d in sorted(p for p in REVIEW.iterdir() if p.is_dir()):
        for f in d.glob("*.jpg"):
            rid = f.name.split("_모델-")[0]
            r = rows[rid]
            seen.add(rid)
            after = "exclude" if d.name == "_제외" else d.name
            out.append(dict(file=r["file"], rid=rid, before=rough_label(r, ant), after=after))
    missing = set(rows) - seen
    assert not missing, f"검수 폴더에서 사라진 파일 {len(missing)}개: {sorted(missing)[:5]}"
    with open("results/relabel_trainval/reviewed_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("검수", len(out), "개 / 바뀜", sum(o["before"] != o["after"] for o in out))
    print("검수 뒤", dict(collections.Counter(o["after"] for o in out)))
    t = collections.Counter((o["before"], o["after"]) for o in out if o["before"] != o["after"])
    for (a, b), n in t.most_common():
        print(f"  {a:14s} → {b:14s} {n}")


if __name__ == "__main__":
    main()

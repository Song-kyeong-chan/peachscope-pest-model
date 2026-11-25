"""재분류 v2 판정을 폴더로 펼친다 (사용자 확인용).

~/peachscope-data/relabel_v2/review/
  healthy/ spot/ rot/ anthracnose/ insect_damage/   ← Claude 판정
  _애매/        ← 원인·병명이 애매한 것 (사용자 판단)
  _애매_희미/   ← 병반이 거의 안 보이는 것 (사용자 판단, 묶어서 결정 가능)
  _제외/        ← 흐림·어두움·과실 아님
파일 이름: <원래 크롭 이름 확장자 제외>__전-<이전 라벨>.jpg  (이름을 바꾸지 말고 폴더만 옮긴다)
"""
import csv
import glob
import shutil
from pathlib import Path

H = Path.home() / "peachscope-data"
SRC = H / "exp3_crops/images"
OUT = H / "relabel_v2/review"
FOLDER = {"?": "_애매", "?f": "_애매_희미", "X": "_제외"}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ["healthy", "spot", "rot", "anthracnose", "insect_damage", *FOLDER.values()]:
        (OUT / d).mkdir(parents=True)
    n = 0
    for p in sorted(glob.glob("results/relabel_v2/calls_*.csv")):
        for r in csv.DictReader(open(p)):
            d = FOLDER.get(r["claude"], r["claude"])
            stem = Path(r["file"]).stem
            shutil.copy(SRC / r["file"], OUT / d / f"{stem}__전-{r['before']}.jpg"); n += 1
    print(n, {d.name: len(list(d.glob('*.jpg'))) for d in sorted(OUT.iterdir())})


if __name__ == "__main__":
    main()

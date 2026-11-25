"""재분류 v2: 5클래스(정상·반점병·부패병·탄저병·충해과)로 학습 데이터 전체를 다시 본다.

- 출발점: results/exp6/labels.csv (팀 검수 + 잿빛무늬병 점검 반영, train+val)
- 이미 하나씩 본 train 잿빛무늬병(br_check.csv)은 다시 보지 않는다.
- 클래스별로 48칸 시트를 만든다. 칸 번호는 그룹 안 순번이다.
출력: ~/peachscope-data/relabel_v2/sheets/<그룹>_NN.jpg, results/relabel_v2/index_<그룹>.csv
"""
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

H = Path.home() / "peachscope-data"
SRC = H / "exp3_crops/images"
OUT = H / "relabel_v2/sheets"
T, G, PER = 170, 8, 48


def main():
    done = {r["file"] for r in csv.DictReader(open("results/relabel_trainval/br_check.csv"))}
    groups = {}
    for r in csv.DictReader(open("results/exp6/labels.csv")):
        if r["split"] == "test":
            continue
        name = Path(r["file"]).name
        if r["cls"] == "rot" and name in done:
            continue
        groups.setdefault(r["cls"], []).append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    Path("results/relabel_v2").mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 15)
    for g, rows in groups.items():
        rows.sort(key=lambda r: Path(r["file"]).name)
        with open(f"results/relabel_v2/index_{g}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["idx", "file", "split", "cls"])
            for i, r in enumerate(rows):
                w.writerow([i, r["file"], r["split"], r["cls"]])
        for s in range(0, len(rows), PER):
            part = rows[s:s + PER]
            sh = Image.new("RGB", (T * G, T * ((len(part) + G - 1) // G)), "white"); d = ImageDraw.Draw(sh)
            for j, r in enumerate(part):
                im = Image.open(SRC / r["file"]).convert("RGB"); im.thumbnail((T - 4, T - 4))
                x, y = (j % G) * T + 2, (j // G) * T + 2
                sh.paste(im, (x, y)); d.rectangle([x, y, x + 40, y + 17], fill="black"); d.text((x + 2, y), str(s + j), fill="yellow", font=font)
            sh.save(OUT / f"{g}_{s // PER:02d}.jpg", quality=85)
        print(g, len(rows), "sheets", (len(rows) + PER - 1) // PER)


if __name__ == "__main__":
    main()

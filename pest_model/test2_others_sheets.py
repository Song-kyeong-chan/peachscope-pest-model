"""test2(팀 검수 test 정답지)의 others 박스 149개를 다시 판정하기 위한 시트.

- results/exp7/labels.csv에서 split=test2, orig=others인 크롭을 모은다.
- 작은 반점을 놓치지 않게 칸을 300px로 크게 한다(4×4=16칸).
출력: ~/peachscope-data/test2_others/sheets/others_NN.jpg, results/test2_others/index.csv (idx, file, d_pred, d_conf)
"""
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

H = Path.home() / "peachscope-data"
SRC = H / "exp3_crops/images"
OUT = H / "test2_others/sheets"
T, G, PER = 300, 4, 16


def main():
    rows = sorted((r for r in csv.DictReader(open("results/exp7/labels.csv")) if r["split"] == "test2" and r["orig"] == "others"),
                  key=lambda r: r["file"])
    pred = {r["file"]: r for r in csv.DictReader(open("results/exp7_D/preds.csv")) if r["split"] == "test2"}
    OUT.mkdir(parents=True, exist_ok=True)
    Path("results/test2_others").mkdir(parents=True, exist_ok=True)
    with open("results/test2_others/index.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx", "file", "d_pred", "d_conf"])
        for i, r in enumerate(rows):
            w.writerow([i, r["file"], pred[r["file"]]["pred"], pred[r["file"]]["conf"]])
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
    for s in range(0, len(rows), PER):
        part = rows[s:s + PER]
        sh = Image.new("RGB", (T * G, T * ((len(part) + G - 1) // G)), "white"); d = ImageDraw.Draw(sh)
        for j, r in enumerate(part):
            im = Image.open(SRC / r["file"]).convert("RGB")
            k = (T - 4) / max(im.size); im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))))
            x, y = (j % G) * T + 2, (j // G) * T + 2
            sh.paste(im, (x, y)); d.rectangle([x, y, x + 46, y + 24], fill="black"); d.text((x + 3, y + 1), str(s + j), fill="yellow", font=font)
        sh.save(OUT / f"others_{s // PER:02d}.jpg", quality=88)
    print(len(rows), "crops,", (len(rows) + PER - 1) // PER, "sheets")


if __name__ == "__main__":
    main()

"""불확실(?) 라벨 2차 판정용 시트: 과실 크롭을 크게(280px) 4×4로 모은다.

- 대상: fruit_boxes.json에서 unsure=True인 사진의 모든 과실 크롭
- 현재 라벨 클래스별로 묶어 시트를 만든다(같은 질문을 연속으로 보게).
- 칸 라벨: 크롭 id(n.k)
출력: ~/peachscope-data/relabel_trainval/usheets/<cls>_NN.jpg, usheets/index.json
2차 판정 결과는 pest_model/results/relabel_trainval/second_pass.txt 에 적는다.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

H = Path.home() / "peachscope-data/relabel_trainval"
S, G = 280, 4


def main():
    fb = json.load(open(H / "fruit_boxes.json"))
    by = {}
    for n, v in sorted(fb.items(), key=lambda x: int(x[0])):
        if not v["unsure"]:
            continue
        for k in range(len(v["boxes"])):
            fp = H / "crops" / v["cls"] / f"{int(n):04d}_{k}.jpg"
            if fp.exists():
                by.setdefault(v["cls"], []).append((f"{n}.{k}", fp))
    out = H / "usheets"; out.mkdir(exist_ok=True)
    F = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
    index = {}
    for cl, lst in by.items():
        for s in range(0, len(lst), G * G):
            name = f"{cl}_{s // (G * G) + 1:02d}"
            sh = Image.new("RGB", (S * G, S * G), "white"); d = ImageDraw.Draw(sh)
            for j, (cid, fp) in enumerate(lst[s:s + G * G]):
                t = Image.open(fp).convert("RGB"); t.thumbnail((S - 6, S - 6), Image.LANCZOS)
                x, y = (j % G) * S + 3, (j // G) * S + 3
                sh.paste(t, (x, y))
                d.rectangle([x, y, x + 84, y + 24], fill="black")
                d.text((x + 3, y + 1), cid, fill="yellow", font=F)
            sh.save(out / f"{name}.jpg", quality=88)
            index[name] = [c for c, _ in lst[s:s + G * G]]
    json.dump(index, open(out / "index.json", "w"), indent=0)
    print({cl: len(v) for cl, v in by.items()}, "sheets", len(index))


if __name__ == "__main__":
    main()

"""train/valid 재분류 1차: 원본(증강 제외) 사진을 4x4 시트로 묶어 사진 단위 분류용으로 만든다.

빨간 박스 = Islamia 원래 라벨(약어). 사진 번호는 order.json 순서.
출력: ~/peachscope-data/relabel_trainval/{order.json, sheets/t###.jpg}
"""
import csv, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

H = Path.home() / "peachscope-data"
OUT = H / "relabel_trainval"
(OUT / "sheets").mkdir(parents=True, exist_ok=True)
NAMES = ["BR", "H", "SH"]  # islamia data.yaml: brown_rot, healty_Peach, shot_hole

rows = list(csv.DictReader(open(H / "islamia/splits.csv")))
groups = {}
for r in rows:
    if r["split"] in ("train", "valid"):
        groups.setdefault(r["group"], r)  # 그룹 대표 1장 (증강 사본 제외)
order = sorted(groups.values(), key=lambda r: (r["split"], r["group_class"], r["group"]))


def boxes(img_path):
    lp = Path(img_path.replace("/images/", "/labels/")).with_suffix(".txt")
    out = []
    if not lp.exists():
        return out
    for l in lp.read_text().split("\n"):
        q = l.split()
        if len(q) == 5:
            c, x, y, w, h = int(q[0]), *map(float, q[1:])
            out.append((c, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        elif len(q) > 5:  # polygon
            c, pts = int(q[0]), list(map(float, q[1:]))
            xs, ys = pts[0::2], pts[1::2]
            out.append((c, min(xs), min(ys), max(xs), max(ys)))
    return out


try:
    F = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
    FS = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
except Exception:
    F = FS = ImageFont.load_default()

S, G = 450, 4
meta = []
for k in range(0, len(order), G * G):
    sheet = Image.new("RGB", (S * G, S * G), "white")
    for j, r in enumerate(order[k:k + G * G]):
        n = k + j + 1
        im = Image.open(r["path"]).convert("RGB")
        W, Hh = im.size
        d = ImageDraw.Draw(im)
        bx = boxes(r["path"])
        for c, x0, y0, x1, y1 in bx:
            d.rectangle([x0 * W, y0 * Hh, x1 * W, y1 * Hh], outline=(255, 40, 40), width=max(2, W // 200))
            d.text((x0 * W + 4, y0 * Hh + 2), NAMES[c], fill=(255, 40, 40), font=F)
        im.thumbnail((S - 8, S - 8))
        sheet.paste(im, ((j % G) * S + 4, (j // G) * S + 4))
        dd = ImageDraw.Draw(sheet)
        tag = f"{n} {NAMES[['brown_rot','healty_Peach','shot_hole'].index(r['group_class'])]}"
        dd.rectangle([(j % G) * S + 4, (j // G) * S + 4, (j % G) * S + 4 + 15 * len(tag), (j // G) * S + 32], fill="black")
        dd.text(((j % G) * S + 8, (j // G) * S + 5), tag, fill="yellow", font=FS)
        meta.append({"n": n, "group": r["group"], "split": r["split"], "orig": r["group_class"],
                     "path": r["path"], "nbox": len(bx), "size": [W, Hh]})
    sheet.save(OUT / "sheets" / f"t{k // (G * G) + 1:03d}.jpg", quality=85)
json.dump(meta, open(OUT / "order.json", "w"), ensure_ascii=False, indent=0)
print(len(meta), "images,", (len(meta) + G * G - 1) // (G * G), "sheets")

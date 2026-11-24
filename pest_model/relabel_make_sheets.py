import json, glob, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLOWorld
H = Path.home() / "peachscope-data"
T = H / "islamia_yolo_disease/test"
D = H / "relabel_draft"
NAMES = ["brown_rot", "healthy", "shot_hole"]
AB = {"brown_rot": "BR", "healthy": "H", "shot_hole": "SH", "others": "OT"}
touched = {v[0]: v[1] for v in json.load(open(D / "touched_backup.json")).values()}
imgs = sorted(T.glob("images/*"))
base = lambda p: p.name.split("_jpg.rf.")[0]
# 현재 라벨 (픽셀 xywh 중심)
cur = {}
for p in imgs:
    b = base(p)
    if b in touched:
        cur[b] = [(c, x, y, w, h) for c, x, y, w, h in touched[b]]
    else:
        L = []
        for l in (T / "labels" / (p.stem + ".txt")).read_text().split("\n"):
            q = l.split()
            if len(q) == 5:
                c, x, y, w, h = int(q[0]), *map(float, q[1:])
                L.append((NAMES[c], x * 640, y * 640, w * 640, h * 640))
        cur[b] = L
m = YOLOWorld("yolov8l-worldv2.pt"); m.set_classes(["peach", "fruit"])
prop = {}
for p in imgs:
    r = m.predict(str(p), conf=0.05, iou=0.5, verbose=False)[0]
    prop[base(p)] = [[round(v, 1) for v in xyxy] + [round(float(s), 2)] for xyxy, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())][:6]
json.dump({"current": cur, "proposals": prop, "order": [base(p) for p in imgs], "touched": sorted(touched)},
          open(D / "draft_input.json", "w"), ensure_ascii=False, indent=0)
try:
    F = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    FS = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
except Exception:
    F = FS = ImageFont.load_default()
S, G = 520, 3
order = [base(p) for p in imgs]
path = {base(p): p for p in imgs}
for k in range(0, len(order), G * G):
    sheet = Image.new("RGB", (S * G, S * G), "white")
    for j, b in enumerate(order[k:k + G * G]):
        im = Image.open(path[b]).convert("RGB")
        d = ImageDraw.Draw(im)
        for i, (x0, y0, x1, y1, s) in enumerate(prop[b]):
            d.rectangle([x0, y0, x1, y1], outline=(0, 230, 255), width=3)
            d.text((x0 + 4, y1 - 26), f"p{i+1}", fill=(0, 230, 255), font=F)
        for i, (c, x, y, w, h) in enumerate(cur[b]):
            d.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline=(255, 40, 40), width=3)
            d.text((x - w / 2 + 4, y - h / 2 + 2), f"o{i+1} {AB[c]}", fill=(255, 40, 40), font=F)
        im = im.resize((S - 8, S - 8))
        sheet.paste(im, ((j % G) * S + 4, (j // G) * S + 4))
        dd = ImageDraw.Draw(sheet)
        tag = f"{k+j+1} {b[:26]}" + (" *" if b in touched else "")
        dd.rectangle([(j % G) * S + 4, (j // G) * S + 4, (j % G) * S + 4 + 14 * len(tag), (j // G) * S + 34], fill="black")
        dd.text(((j % G) * S + 8, (j // G) * S + 6), tag, fill="yellow", font=FS)
    sheet.save(D / "sheets" / f"s{k // (G*G) + 1:02d}.jpg", quality=88)
print(len(order), "images,", len(os.listdir(D / "sheets")), "sheets")

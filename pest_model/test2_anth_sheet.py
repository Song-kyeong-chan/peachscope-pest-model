"""test2 v2 탄저병 정답 15개를 모델 9개(seed 3 × 3)의 예측과 함께 4×4 시트로 만든다.
원본 사진 전체(박스 표시)와 크롭을 나란히 둔다. 출력: ~/peachscope-data/test2_anth/sheet_*.jpg, results/test2_others/anth_index.csv"""
import collections, csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
H = Path.home() / "peachscope-data"; IMG = H / "exp3_crops/images"; OUT = H / "test2_anth"
RUNS = [f"exp7_{m}{s}" for m in ("D", "ks", "clip_ft") for s in ("", "_s1", "_s2")]
AB = {"healthy": "H", "spot": "S", "rot": "R", "anthracnose": "A", "damage": "D"}
rows = [r for r in csv.DictReader(open("results/test2_others/labels_v2.csv")) if r["cls_v2"] == "anthracnose"]
pred = {run: {r["file"]: r["pred"] for r in csv.DictReader(open(f"results/{run}/preds.csv")) if r["split"] == "test2"} for run in RUNS}
idx = []
tiles = []
for i, r in enumerate(rows):
    ps = [pred[run][r["file"]] for run in RUNS]
    c = collections.Counter(ps)
    im = ImageOps.exif_transpose(Image.open(IMG / r["file"])).convert("RGB")
    im.thumbnail((400, 400)); t = Image.new("RGB", (400, 440), "white"); t.paste(im, ((400 - im.width) // 2, 40))
    d = ImageDraw.Draw(t); d.text((5, 5), f"#{i:02d} orig={r['orig']}  " + " ".join(f"{AB[k]}{v}" for k, v in c.most_common()), fill="black")
    tiles.append(t)
    idx.append(dict(idx=i, file=r["file"], orig=r["orig"], preds=" ".join(AB[p] for p in ps), n_anth=c["anthracnose"]))
for s in range(0, len(tiles), 8):
    sh = Image.new("RGB", (1600, 880), "white")
    for k, t in enumerate(tiles[s:s + 8]):
        sh.paste(t, ((k % 4) * 400, (k // 4) * 440))
    sh.save(OUT / f"sheet_{s // 8}.jpg", quality=90)
with open("results/test2_others/anth_index.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(idx[0])); w.writeheader(); w.writerows(idx)
for x in idx: print(x["idx"], x["orig"], x["preds"], x["file"][:60])

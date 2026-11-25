"""실험 7 라벨: 재분류 v2(review 폴더) 반영 + 구멍·상처(damage) 5클래스.

- 출발점: results/exp6/labels.csv (팀 검수 + 잿빛무늬병 점검 반영)
- ~/peachscope-data/relabel_v2/review/<폴더>/<크롭 stem>__전-<이전>.jpg 가 있으면 그 폴더를 라벨로 쓴다.
  insect_damage → damage, _애매·_애매_희미·_제외 → 학습에서 뺀다. review에 없는 크롭은 exp6 라벨(insect_damage → damage).
- test(NCPMS): 원래 병명 기준 brown_rot·gray_mold → rot, shot_hole·scab → spot, anthracnose, insect_damage·stink_bug → damage.
  planthopper(난괴)·viroid는 5클래스에 해당 없어 뺀다.
- test2(팀 검수 test 정답지, Roboflow peach-test-relabel v1): 정답 박스를 10% 넓혀 잘라 images/test2/에 저장.
  shot_hole·spot → spot, brown_rot·others → rot, insect_damage → damage.
출력: results/exp7/labels.csv (file, split, cls, status, orig)
"""
import collections
import csv
import json
from pathlib import Path

import yaml
from PIL import Image, ImageOps

H = Path.home() / "peachscope-data"
IMG = H / "exp3_crops/images"
REVIEW = H / "relabel_v2/review"
TEST2 = H / "relabel_test_v1"
FOLDER = {"healthy": "healthy", "spot": "spot", "rot": "rot", "anthracnose": "anthracnose", "insect_damage": "damage"}
NCPMS7 = {"brown_rot": "rot", "gray_mold": "rot", "shot_hole": "spot", "scab": "spot", "anthracnose": "anthracnose",
          "insect_damage": "damage", "stink_bug": "damage"}
TEST2_MAP = {"shot_hole": "spot", "spot": "spot", "brown_rot": "rot", "others": "rot", "anthracnose": "anthracnose",
             "healthy": "healthy", "insect_damage": "damage"}


def review_labels():
    out = {}
    for d in REVIEW.iterdir():
        if not d.is_dir():
            continue
        for f in d.glob("*.jpg"):
            out[f.name.split("__전-")[0]] = FOLDER.get(d.name, "exclude")
    return out


def build_test2(rows):
    names = yaml.safe_load(open(TEST2 / "data.yaml"))["names"]
    dst = IMG / "test2"; dst.mkdir(exist_ok=True)
    for lp in sorted((TEST2 / "test/labels").glob("*.txt")):
        ip = next((TEST2 / "test/images").glob(lp.stem + ".*"))
        im = ImageOps.exif_transpose(Image.open(ip)).convert("RGB"); W, Hh = im.size
        for k, l in enumerate(x for x in lp.read_text().split("\n") if x.strip()):
            c, x, y, w, h = l.split()[:5]
            x, y, w, h = float(x) * W, float(y) * Hh, float(w) * W, float(h) * Hh
            pw, ph = w * .1, h * .1
            box = (max(0, x - w / 2 - pw), max(0, y - h / 2 - ph), min(W, x + w / 2 + pw), min(Hh, y + h / 2 + ph))
            name = f"t2_{lp.stem[:40]}_{k}.jpg"
            if not (dst / name).exists():
                im.crop(box).save(dst / name, quality=92)
            rows.append(dict(file=f"test2/{name}", split="test2", cls=TEST2_MAP[names[int(c)]], status="test2", orig=names[int(c)]))


def main():
    rv = review_labels()
    rows, c = [], collections.Counter()
    for r in csv.DictReader(open("results/exp6/labels.csv")):
        if r["split"] == "test":
            continue
        stem = Path(r["file"]).stem
        if stem in rv:
            cls, st = rv[stem], "relabel_v2"
        else:
            cls, st = ("damage" if r["cls"] == "insect_damage" else r["cls"]), r["status"]
        c[st == "relabel_v2"] += 1
        if cls == "exclude":
            continue
        rows.append(dict(file=r["file"], split=r["split"], cls=cls, status=st, orig=r["orig"]))
    orig = {Path(e["file"]).stem: e["cls"] for e in json.load(open(H / "ncpms/fruit_images.json"))}
    for r in csv.DictReader(open(H / "exp3_crops/splits.csv")):
        if r["split"] == "test":
            o = orig[Path(r["file"]).stem[3:]]
            if o in NCPMS7:
                rows.append(dict(file=r["file"], split="test", cls=NCPMS7[o], status="ncpms", orig=o))
    build_test2(rows)
    out = Path("results/exp7"); out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cc = collections.Counter((r["split"], r["cls"]) for r in rows)
    for s in ("train", "val", "test", "test2"):
        d = {k: v for (ss, k), v in sorted(cc.items()) if ss == s}
        print(s, sum(d.values()), d)
    print("review 반영", c[True], "/ exp6 그대로", c[False])


if __name__ == "__main__":
    main()

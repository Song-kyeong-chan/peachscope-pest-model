import json, os, re
from collections import Counter
from pathlib import Path
H = Path.home() / "peachscope-data"
D = H / "relabel_draft"
T = H / "islamia_yolo_disease/test"
inp = json.load(open(D / "draft_input.json"))
order, cur, prop, touched = inp["order"], inp["current"], inp["proposals"], set(inp["touched"])
CL = {"H": "healthy", "SH": "shot_hole", "BR": "brown_rot", "OT": "others"}
NAMES = ["brown_rot", "healthy", "shot_hole", "others"]
dec = {}
for l in open(D / "decisions.txt"):
    m = re.match(r"^(\d+) \| (.*?) \| (.*)$", l.strip())
    if m:
        dec[int(m[1])] = (m[2].split(), m[3])
assert len(dec) == 223
imgs = {p.name.split("_jpg.rf.")[0]: p for p in (T / "images").iterdir()}
(D / "test/images").mkdir(parents=True, exist_ok=True)
(D / "test/labels").mkdir(parents=True, exist_ok=True)
final, fix, cnt, changed = {}, [], Counter(), []
for i, b in enumerate(order, 1):
    toks, note = dec[i]
    boxes = []
    if toks == ["K"]:
        boxes = [(c, x, y, w, h, False) for c, x, y, w, h in cur[b]]
    else:
        for t in toks:
            k, c = t.split("=")
            bang = c.endswith("!")
            c = CL[c.rstrip("!")]
            if k == "F":
                boxes.append((c, 320, 320, 640, 640, bang))
            elif k[0] == "o":
                _, x, y, w, h = cur[b][int(k[1:]) - 1]
                boxes.append((c, x, y, w, h, bang))
            else:
                x0, y0, x1, y1, _ = prop[b][int(k[1:]) - 1]
                boxes.append((c, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, bang))
    for c, *_ , bang in boxes:
        cnt[c] += 1
    if any(bb[-1] for bb in boxes):
        fix.append((i, b))
    old = Counter(c for c, *_ in cur[b])
    new = Counter(c for c, *_r in boxes)
    if b in touched and old != new:
        changed.append((i, b, dict(old), dict(new), note))
    final[b] = [[c, round(x, 2), round(y, 2), round(w, 2), round(h, 2)] for c, x, y, w, h, _ in boxes]
    p = imgs[b]
    dst = D / "test/images" / p.name
    if not dst.exists():
        os.symlink(p, dst)
    (D / "test/labels" / (p.stem + ".txt")).write_text("\n".join(
        f"{NAMES.index(c)} {x/640:.6f} {y/640:.6f} {w/640:.6f} {h/640:.6f}" for c, x, y, w, h in final[b]) + "\n")
json.dump(final, open(D / "final_boxes.json", "w"), ensure_ascii=False, indent=0)
(D / "test/data.yaml").write_text("names: [brown_rot, healthy, shot_hole, others]\n")
print("박스 합계", sum(cnt.values()), dict(cnt))
img_cls = Counter()
for b, bx in final.items():
    for c in {x[0] for x in bx}:
        img_cls[c] += 1
print("클래스가 있는 사진 수", dict(img_cls))
print("박스를 사람이 고쳐야 하는 사진:", fix)
print("사용자가 고친 사진 중 클래스를 바꾼 것:", len(changed))
for c in changed:
    print("  ", c[0], c[1], c[2], "->", c[3])

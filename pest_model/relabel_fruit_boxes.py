"""train/valid 재분류 2차: 사진 단위 1차 분류(decisions.txt)를 과실 단위 박스로 바꾼다.

규칙
- YOLO-World("peach","fruit") 제안 + 실험 2 과실 탐지기(YOLO11n) 박스를 과실 후보로 쓴다.
- Islamia 원래 박스(병반 위치)의 중심을 포함하는 과실 후보에 그 사진의 클래스를 준다.
  원래 박스 자체가 과실 크기(제안과 IoU>0.5)면 원래 박스를 그대로 쓴다.
- 'm'(과실 여러 개)이 아니면 가장 잘 맞는 과실 1개만 라벨한다.
- 증상 박스와 무관한 다른 과실은 라벨하지 않는다(학습에서 제외).
- 코드가 섞인 사진(mixed)은 자동 처리하지 않고 manual.txt로 넘긴다.
- 짧은 변 12% 미만 후보(병반 크기)와 80px 미만 크롭은 버린다.
- X(사용불가), NP(복숭아 아님)는 제외.
출력: ~/peachscope-data/relabel_trainval/{fruit_boxes.json, crops/<class>/*.jpg, csheets/*.jpg, manual.txt}
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO, YOLOWorld

H = Path.home() / "peachscope-data/relabel_trainval"
CLS = {"H": "healthy", "SH": "shot_hole", "SC": "scab", "BR": "brown_rot", "INS": "insect_damage",
       "ANT": "others", "ROT": "others", "PM": "others"}
meta = {e["n"]: e for e in json.load(open(H / "order.json"))}


def parse():
    dec = {}
    for l in open(H / "decisions.txt"):
        l = l.split("#")[0].split()
        if l:
            dec[int(l[0])] = (l[1].split(","), l[2:])
    return dec


def orig_boxes(path, W, Hh):
    lp = Path(path.replace("/images/", "/labels/")).with_suffix(".txt")
    out = []
    for l in (lp.read_text().split("\n") if lp.exists() else []):
        q = l.split()
        if len(q) == 5:
            _, x, y, w, h = int(q[0]), *map(float, q[1:])
            out.append([(x - w / 2) * W, (y - h / 2) * Hh, (x + w / 2) * W, (y + h / 2) * Hh])
        elif len(q) > 5:
            p = list(map(float, q[1:]))
            out.append([min(p[0::2]) * W, min(p[1::2]) * Hh, max(p[0::2]) * W, max(p[1::2]) * Hh])
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def contains(f, o):
    cx, cy = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
    return f[0] <= cx <= f[2] and f[1] <= cy <= f[3]


def main():
    dec = parse()
    m = YOLOWorld("yolov8l-worldv2.pt"); m.set_classes(["peach", "fruit"])
    fd = YOLO("results/exp2_yolo/yolo11n_fruit/weights/best.pt")  # 실험 2 과실 탐지기 (보조)
    res, manual = {}, []
    for n, (codes, flags) in sorted(dec.items()):
        if codes[0] in ("X", "NP"):
            continue
        if len(set(codes)) > 1:
            manual.append(n); continue
        cls = CLS[codes[0]]
        e = meta[n]
        im = Image.open(e["path"]); W, Hh = im.size
        r = m.predict(e["path"], conf=0.1, iou=0.5, verbose=False)[0]
        props = [b for b in r.boxes.xyxy.tolist()]
        confs = r.boxes.conf.tolist()
        r2 = fd.predict(e["path"], conf=0.25, verbose=False)[0]
        for b, c in zip(r2.boxes.xyxy.tolist(), r2.boxes.conf.tolist()):
            if all(iou(b, q) < 0.6 for q in props):
                props.append(b); confs.append(c * 0.5)
        # 병반 크기 박스 제거: 과실 후보는 짧은 변이 사진 짧은 변의 12% 이상
        keep = [i for i, b in enumerate(props) if min(b[2] - b[0], b[3] - b[1]) >= 0.12 * min(W, Hh)]
        props = [props[i] for i in keep]; confs = [confs[i] for i in keep]
        O = orig_boxes(e["path"], W, Hh)
        boxes = []
        for o in O:
            # 원래 박스가 과실 크기면 그대로
            best = max(range(len(props)), key=lambda i: iou(props[i], o), default=None)
            if best is not None and iou(props[best], o) > 0.5:
                boxes.append(props[best]); continue
            cand = [i for i in range(len(props)) if contains(props[i], o)]
            if cand:  # 증상을 포함하는 가장 작은 과실 후보 (겹친 큰 박스 방지)
                i = min(cand, key=lambda i: (props[i][2] - props[i][0]) * (props[i][3] - props[i][1]))
                boxes.append(props[i])
            elif (o[2] - o[0]) * (o[3] - o[1]) > 0.1 * W * Hh:
                boxes.append(o)  # 제안이 없고 원래 박스가 큰 경우
        # 중복 제거
        uniq = []
        for b in boxes:
            if all(iou(b, u) < 0.7 for u in uniq):
                uniq.append(b)
        if not uniq and props:  # 원래 박스와 맞는 후보가 없으면 최고 신뢰 후보
            uniq = [props[max(range(len(props)), key=lambda i: confs[i])]]
        if "m" not in flags and len(uniq) > 1:
            # 단일 과실 사진: 원래 박스 중심을 가장 많이 포함하는 것 1개
            uniq = [max(uniq, key=lambda f: (sum(contains(f, o) for o in O), (f[2] - f[0]) * (f[3] - f[1])))]
        uniq = [b for b in uniq if min(b[2] - b[0], b[3] - b[1]) >= 80]  # 너무 작은 크롭은 버림
        res[n] = {"cls": cls, "unsure": "?" in flags, "wm": "t" in flags, "split": e["split"],
                  "path": e["path"], "boxes": [[round(v, 1) for v in b] for b in uniq]}
    json.dump(res, open(H / "fruit_boxes.json", "w"), ensure_ascii=False, indent=0)
    (H / "manual.txt").write_text("\n".join(map(str, manual)) + "\n")
    # 크롭 저장 + 검토용 시트
    crops = []
    for n, v in res.items():
        im = Image.open(v["path"]).convert("RGB")
        for k, (x0, y0, x1, y1) in enumerate(v["boxes"]):
            pw, ph = (x1 - x0) * 0.1, (y1 - y0) * 0.1
            c = im.crop((max(0, x0 - pw), max(0, y0 - ph), min(im.width, x1 + pw), min(im.height, y1 + ph)))
            d = H / "crops" / v["cls"]; d.mkdir(parents=True, exist_ok=True)
            fp = d / f"{n:04d}_{k}.jpg"; c.save(fp, quality=90)
            crops.append((v["cls"], n, k, fp, v["unsure"]))
    F = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
    (H / "csheets").mkdir(exist_ok=True)
    S, G = 225, 8
    by = {}
    for c in crops:
        by.setdefault(c[0], []).append(c)
    for cl, lst in by.items():
        for s in range(0, len(lst), G * G):
            sh = Image.new("RGB", (S * G, S * G), "white"); dd = ImageDraw.Draw(sh)
            for j, (_, n, k, fp, un) in enumerate(lst[s:s + G * G]):
                t = Image.open(fp); t.thumbnail((S - 6, S - 6))
                x, y = (j % G) * S + 3, (j // G) * S + 3
                sh.paste(t, (x, y))
                dd.rectangle([x, y, x + 78, y + 22], fill="black")
                dd.text((x + 3, y + 1), f"{n}.{k}{'?' if un else ''}", fill="yellow", font=F)
            sh.save(H / "csheets" / f"{cl}_{s // (G * G) + 1:02d}.jpg", quality=85)
    import collections
    print("images", len(res), "manual", len(manual))
    print("crops", dict(collections.Counter(c[0] for c in crops)))
    print("no-box images", [n for n, v in res.items() if not v["boxes"]])


if __name__ == "__main__":
    main()

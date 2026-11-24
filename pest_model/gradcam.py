"""Grad-CAM: 모델이 사진의 어디를 보고 판단했는지 확인한다.

test 사진을 클래스별로 몇 장씩 골라 원본과 열지도를 나란히 붙인 한 장의 이미지를 만든다.
열지도가 과실 병반 위에 있으면 병반을 학습한 것이고, 배경·잎·하늘에 있으면 지름길 학습이다.
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T

CLASSES = ["healthy", "shot_hole", "brown_rot"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TARGET_LAYER = {"resnet50": "layer4", "efficientnet_b0": "conv_head",
                "mobilenetv3_large_100": "blocks", "convnext_tiny": "stages"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--per_class", type=int, default=6)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    model = timm.create_model(args.model, pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(Path(args.out) / f"{args.model}.pt", map_location="cpu"))
    model.eval()
    layer = model.get_submodule(TARGET_LAYER[args.model])
    acts, grads = {}, {}
    layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))

    tf_vis = T.Compose([T.Resize(args.size), T.CenterCrop(args.size)])
    norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    rows = [r for r in csv.DictReader(open(Path(args.data) / "splits.csv")) if r["split"] == "test"]
    rng = random.Random(0)
    picked = []
    for c in CLASSES:
        rs = [r for r in rows if r["cls"] == c]
        picked += rng.sample(rs, min(args.per_class, len(rs)))

    S = args.size
    sheet = Image.new("RGB", (S * 2 * 3, (S + 18) * ((len(picked) + 2) // 3)), "white")
    for k, r in enumerate(picked):
        im = tf_vis(Image.open(Path(args.data) / "images" / r["file"]).convert("RGB"))
        x = norm(im)[None]
        model.zero_grad()
        logits = model(x)
        pred = int(logits.argmax(1))
        logits[0, pred].backward()
        w = grads["g"].mean((2, 3), keepdim=True)
        cam = torch.relu((w * acts["a"]).sum(1))[0].detach().numpy()
        cam = cam / (cam.max() + 1e-8)
        cam = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((S, S), Image.BILINEAR)) / 255.0
        heat = np.zeros((S, S, 3))
        heat[..., 0] = np.clip(cam * 2, 0, 1)
        heat[..., 1] = np.clip(cam * 2 - 1, 0, 1)
        over = np.uint8(np.asarray(im) * 0.5 + heat * 255 * 0.5)
        col, row = k % 3, k // 3
        x0, y0 = col * S * 2, row * (S + 18)
        sheet.paste(im, (x0, y0 + 18))
        sheet.paste(Image.fromarray(over), (x0 + S, y0 + 18))
        ok = "O" if CLASSES[pred] == r["cls"] else "X"
        ImageDraw.Draw(sheet).text((x0 + 4, y0 + 3), f"[{ok}] true={r['cls']}  pred={CLASSES[pred]}", fill="black")
    dst = Path(args.out) / f"gradcam_{args.model}.jpg"
    sheet.save(dst, quality=88)
    print("saved", dst)


if __name__ == "__main__":
    main()

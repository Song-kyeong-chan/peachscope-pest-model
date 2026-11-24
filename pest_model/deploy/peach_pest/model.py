"""ConvNeXt-Tiny 병해충 분류기 (실험 4, 6클래스) + 선택적 과실 탐지기(YOLO11n).

전처리는 학습과 반드시 같아야 한다.
  EXIF 회전 반영 → RGB → 과실 박스를 가로·세로 10% 넓혀 자르기
  → 짧은 변 288px로 리사이즈 → 가운데 288×288 → ImageNet 정규화

과실이 없는 사진을 판정하지 않기 위한 장치 (2026-11-23)
  - 과실을 찾지 못하면 사진 전체를 판정하지 않고 status="no_fruit"를 돌려준다.
  - 탐지 박스는 확신도 0.25 이상, 짧은 변이 사진 짧은 변의 10% 이상이면서 64px 이상인 것만 쓴다.
  - 잘라 낸 과실이 거의 단색(밝기 표준편차 6 미만)이면 버린다. 실제 과실 크롭 4,134개의 최솟값은 9.2였다.
  이 장치로 단색·빈 사진은 걸러지지만, 잎·땅처럼 무늬가 있는 사진은 탐지기가 과실로 잡으면 여전히 판정된다.
"""
from io import BytesIO
from pathlib import Path
from typing import Optional

import timm
import torch
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError
from torchvision import transforms as T

from .base import Box, FruitResult, PestResult
from .guide import TIER_ORDER, Guide

CLASSES = ["healthy", "spot", "brown_rot", "anthracnose", "insect_damage", "others"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SIZE = 288
PAD = 0.10
UNCERTAIN_THR = 0.5   # 임시값. 팀 검수 후 검증 데이터로 다시 정한다
DET_CONF = 0.25       # 0.4로 올리면 실제 사진 60장 중 과실 못 찾음이 1→5장으로 늘고, 단색 사진(박스 확신도 0.74)은 못 막아서 0.25 유지
MIN_SIDE_RATIO = 0.10  # 정답 박스 짧은 변 비율의 하위 1%가 0.147
MIN_SIDE_PX = 64
MIN_TEXTURE = 6.0     # 128×128 흑백 크롭의 밝기 표준편차
NO_FRUIT_MESSAGE = "사진에서 과실을 찾지 못했습니다. 봉지를 벗긴 과실 하나가 화면을 크게 채우도록 밝은 곳에서 다시 찍어 주세요."


class InvalidImageError(ValueError):
    pass


class ConvNeXtPestModel:
    version = "pest-exp4-convnext-2026-11-23"

    def __init__(self, weights: Path, guide: Path, detector: Optional[Path] = None, device: str = "cpu",
                 uncertain_thr: float = UNCERTAIN_THR, det_conf: float = DET_CONF, allow_whole_image: bool = False):
        self.device = device
        self.net = timm.create_model("convnext_tiny", pretrained=False, num_classes=len(CLASSES))
        self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.net.eval().to(device)
        self.tf = T.Compose([T.Resize(SIZE), T.CenterCrop(SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])
        self.guide = Guide(guide)
        self.thr = uncertain_thr
        self.det_conf = det_conf
        self.allow_whole_image = allow_whole_image  # 테스트용. 서버에서는 False로 둔다
        self.detector = None
        if detector:
            from ultralytics import YOLO  # 탐지기를 쓸 때만 필요
            self.detector = YOLO(str(detector))

    @staticmethod
    def load_image(image_bytes: bytes) -> Image.Image:
        try:
            return ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise InvalidImageError("이미지를 읽을 수 없습니다") from e

    def detect(self, img: Image.Image) -> list[Box]:
        r = self.detector.predict(img, conf=self.det_conf, verbose=False)[0]
        return [tuple(map(float, b)) for b in r.boxes.xyxy.tolist()]

    @staticmethod
    def box_ok(img: Image.Image, box: Box) -> bool:
        side = min(box[2] - box[0], box[3] - box[1])
        return side >= MIN_SIDE_PX and side >= MIN_SIDE_RATIO * min(img.size)

    @staticmethod
    def crop(img: Image.Image, box: Box) -> Image.Image:
        W, H = img.size
        x1, y1, x2, y2 = box
        pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
        return img.crop((max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph)))

    @staticmethod
    def textured(crop: Image.Image) -> bool:
        return ImageStat.Stat(crop.convert("L").resize((128, 128))).stddev[0] >= MIN_TEXTURE

    @torch.no_grad()
    def classify(self, crops: list[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.tf(c) for c in crops]).to(self.device)
        return self.net(x).softmax(1).cpu()

    def no_fruit(self, skipped: int = 0) -> PestResult:
        return PestResult(fruits=[], summary="no_fruit", summary_guide_id="", summary_tier="none",
                          model_version=self.version, status="no_fruit", message=NO_FRUIT_MESSAGE,
                          skipped_boxes=skipped)

    def predict(self, image_bytes: bytes, boxes: Optional[list[Box]] = None) -> PestResult:
        img = self.load_image(image_bytes)
        src = "given"
        if boxes is None:
            if self.detector:
                boxes, src = self.detect(img), "detector"
            elif self.allow_whole_image:
                boxes, src = [(0.0, 0.0, float(img.width), float(img.height))], "whole_image"
            else:
                boxes = []
        kept = []
        for b in boxes:
            if src == "whole_image" or self.box_ok(img, b):
                c = self.crop(img, b)
                if self.textured(c):
                    kept.append((b, c))
        if not kept:
            return self.no_fruit(len(boxes))
        probs = self.classify([c for _, c in kept])
        fruits = []
        for (b, _), p in zip(kept, probs):
            pd = {c: round(float(v), 4) for c, v in zip(CLASSES, p)}
            label = max(pd, key=pd.get)
            unc = pd[label] < self.thr
            gid, gtype, tier = self.guide.unknown() if unc else self.guide.resolve(label)
            fruits.append(FruitResult(box=tuple(round(v, 1) for v in b), label=label, confidence=pd[label], probs=pd,
                                      uncertain=unc, guide_id=gid, guide_type=gtype, tier=tier,
                                      questions=[] if unc else self.guide.questions_for(pd), box_source=src))
        sick = [f for f in fruits if f.label != "healthy" or f.uncertain]
        if sick:
            worst = max(sick, key=lambda f: (TIER_ORDER[f.tier], f.confidence))
            summary = (worst.label, worst.guide_id, worst.tier)
        else:
            gid, _, tier = self.guide.resolve("healthy")
            summary = ("healthy", gid, tier)
        return PestResult(fruits=fruits, summary=summary[0], summary_guide_id=summary[1], summary_tier=summary[2],
                          model_version=self.version, skipped_boxes=len(boxes) - len(kept))

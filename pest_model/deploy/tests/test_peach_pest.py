"""가중치가 있을 때만 도는 간단한 점검: cd pest_model/deploy && python -m pytest -q tests"""
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from peach_pest import CLASSES, ConvNeXtPestModel, InvalidImageError

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "weights/pest_convnext_exp7D.pt"
pytestmark = pytest.mark.skipif(not W.exists(), reason="가중치 파일이 없음 (드라이브에서 받아 weights/에 두기)")


@pytest.fixture(scope="module")
def model():
    return ConvNeXtPestModel(W, ROOT / "knowledge/peach_pest_guide.json")


def jpg(color=(200, 120, 90), size=(640, 480)):
    b = BytesIO(); Image.new("RGB", size, color).save(b, "JPEG"); return b.getvalue()


def test_given_boxes(model):
    noisy = Image.effect_noise((640, 480), 60).convert("RGB")
    b = BytesIO(); noisy.save(b, "JPEG")
    r = model.predict(b.getvalue(), boxes=[(10, 10, 200, 200), (300, 100, 500, 400)])
    assert len(r.fruits) == 2 and r.model_version == model.version
    for f in r.fruits:
        assert f.label in CLASSES and abs(sum(f.probs.values()) - 1) < 1e-2
        assert f.tier in ("urgent", "soon", "next_season", "none") and f.box_source == "given"
    assert r.to_dict()["summary"] in CLASSES


def test_no_boxes_no_detector_is_no_fruit(model):
    r = model.predict(jpg())
    assert r.status == "no_fruit" and r.fruits == [] and r.message


def test_blank_crops_rejected(model):
    # 단색 사진은 박스가 있어도 과실로 보지 않는다 (탐지기가 단색 사진에 0.74로 박스를 낸 적이 있음)
    r = model.predict(jpg((120, 80, 50)), boxes=[(100, 100, 500, 500)])
    assert r.status == "no_fruit" and r.skipped_boxes == 1


def test_tiny_box_rejected(model):
    noisy = Image.effect_noise((640, 480), 60).convert("RGB")
    b = BytesIO(); noisy.save(b, "JPEG")
    r = model.predict(b.getvalue(), boxes=[(10, 10, 40, 40), (100, 100, 400, 400)])
    assert len(r.fruits) == 1 and r.skipped_boxes == 1


def test_invalid_image(model):
    with pytest.raises(InvalidImageError):
        model.predict(b"not an image")


def test_guide_mapping(model):
    g = model.guide
    assert g.resolve("others")[:2] == ("rot", "group")
    assert g.resolve("spot")[:2] == ("spot", "group")
    assert g.resolve("brown_rot")[2] == "urgent"
    assert g.questions_for({"spot": 0.5, "insect_damage": 0.4, "healthy": 0.1}) == ["spot_split", "spot_vs_damage"]
    assert g.resolve("insect_damage")[:2] == ("damage", "group")


def test_exp7_classes(model):
    g = model.guide
    assert CLASSES == ["healthy", "spot", "rot", "anthracnose", "damage"]
    assert g.resolve("rot")[:2] == ("rot", "group") and g.resolve("rot")[2] == "urgent"
    assert g.resolve("damage")[:2] == ("damage", "group")
    assert g.questions_for({"damage": 0.7, "spot": 0.2, "rot": 0.1}) == ["damage_split"]

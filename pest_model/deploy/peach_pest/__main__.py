"""사진 한 장 판정: python -m peach_pest <사진> [--weights ...] [--detector ...] [--guide ...]"""
import argparse
import json
from pathlib import Path

from .model import ConvNeXtPestModel

ROOT = Path(__file__).resolve().parents[2]  # pest_model/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", default=str(ROOT / "weights/pest_convnext_exp7D.pt"))
    ap.add_argument("--detector", default=str(ROOT / "weights/fruit_yolo11n.pt"), help="빈 문자열이면 탐지기 없이 사진 전체를 한 과실로 본다")
    ap.add_argument("--whole", action="store_true", help="탐지기 없이 사진 전체를 한 과실로 판정 (테스트용)")
    ap.add_argument("--guide", default=str(ROOT / "knowledge/peach_pest_guide.json"))
    a = ap.parse_args()
    m = ConvNeXtPestModel(Path(a.weights), Path(a.guide), Path(a.detector) if a.detector else None,
                          allow_whole_image=a.whole)
    print(json.dumps(m.predict(Path(a.image).read_bytes()).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

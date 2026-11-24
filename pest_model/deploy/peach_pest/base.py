"""병해충 모델 인터페이스. 서버의 quality/base.py(QualityModel)와 같은 방식이다."""
from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol

Box = tuple[float, float, float, float]  # 원본 사진 픽셀 좌표 (x1, y1, x2, y2)


@dataclass(frozen=True)
class FruitResult:
    box: Box
    label: str                    # 모델 클래스 (healthy, spot, brown_rot, anthracnose, insect_damage, others)
    confidence: float             # 그 클래스의 확률
    probs: dict[str, float]       # 클래스별 확률
    uncertain: bool               # confidence가 기준 미만이면 True → 앱에서 '판별 불가'로 안내
    guide_id: str                 # peach_pest_guide.json의 entries id 또는 display_groups id
    guide_type: str               # "entry" | "group"
    tier: str                     # 대응 등급: urgent | soon | next_season | none
    questions: list = field(default_factory=list)  # 앱에서 물어볼 질문(disambiguation 규칙 id)
    box_source: str = "given"     # given(서버가 준 박스) | detector(내장 탐지기) | whole_image(테스트용 사진 전체)


@dataclass(frozen=True)
class PestResult:
    fruits: list[FruitResult]
    summary: str                  # 가장 급한 병해의 모델 클래스. 모두 정상이면 healthy, 과실이 없으면 no_fruit
    summary_guide_id: str
    summary_tier: str
    model_version: str
    status: str = "ok"            # ok | no_fruit (과실을 찾지 못해 판정하지 않음 → message를 보여 주고 다시 찍기 안내)
    message: str = ""
    skipped_boxes: int = 0        # 너무 작거나 단색이라 버린 박스 수

    def to_dict(self) -> dict:
        return asdict(self)


class PestModel(Protocol):
    version: str

    def predict(self, image_bytes: bytes, boxes: Optional[list[Box]] = None) -> PestResult: ...

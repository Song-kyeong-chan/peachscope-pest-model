# 병해충 모델 서버 연동 패키지 (`peach_pest`)

봉지를 벗긴 미홍 과실 사진을 받아 과실마다 병해충을 판정하고, 대응방안 JSON(`pest_model/knowledge/peach_pest_guide.json`)의 항목과 대응 등급을 붙여 돌려준다. 서버 `quality/base.py`의 `QualityModel`과 같은 Protocol 방식이다.

- 모델 버전: `pest-exp7D-convnext-2026-11-25` (실험 7 D: 재분류 v2 라벨, 5클래스, 지식 스무딩 손실 + 증상 증강)
- 대응방안 JSON: v0.6.0
- 실험 4 가중치(6클래스)에서 바뀐 점: 클래스가 `healthy`, `spot`, `rot`, `anthracnose`, `damage` 5개가 됐다. 인터페이스(입력·출력 형식)는 그대로다.

## 1. 파일

| 파일 | 내용 | 받는 곳 |
|---|---|---|
| `peach_pest/` | 추론 코드 (base, model, guide) | 이 저장소 |
| `../knowledge/peach_pest_guide.json` | 증상·대응·출하·대응 등급·질문 규칙 | 이 저장소 |
| `../weights/pest_convnext_exp7D.pt` | ConvNeXt-Tiny 분류기, 약 106MB | **GitHub Releases** `v0.7-exp7D` (Git 제외) |
| `../weights/fruit_yolo11n.pt` | 과실 탐지기, 약 5MB. 서버가 과실 박스를 직접 주면 없어도 됨 | **GitHub Releases** `v0.7-exp7D` (Git 제외) |

가중치 받기 (저장소 루트에서, 로그인 필요 없음):

```bash
mkdir -p pest_model/weights
curl -L -o pest_model/weights/pest_convnext_exp7D.pt https://github.com/Song-kyeong-chan/peachscope-pest-model/releases/download/v0.7-exp7D/pest_convnext_exp7D.pt
curl -L -o pest_model/weights/fruit_yolo11n.pt https://github.com/Song-kyeong-chan/peachscope-pest-model/releases/download/v0.7-exp7D/fruit_yolo11n.pt
shasum -a 256 pest_model/weights/*.pt   # 릴리스 설명의 SHA-256과 비교
```

웹에서는 저장소의 Releases → `v0.7-exp7D`에서 두 파일을 받아 `pest_model/weights/`에 둔다.

## 2. 설치와 확인

```bash
pip install -r pest_model/deploy/requirements.txt
cd pest_model/deploy
python -m peach_pest 사진.jpg          # 결과 JSON 출력
python -m pytest -q tests              # 가중치가 있으면 7개 통과
```

## 3. 서버에 붙이기 (예시, `server/api/app/main.py`)

```python
from peach_pest import ConvNeXtPestModel, InvalidImageError as PestImageError

P = Path("/models")  # 가중치와 JSON을 둔 곳
pest_model = ConvNeXtPestModel(P / "pest_convnext_exp7D.pt", P / "peach_pest_guide.json",
                               detector=P / "fruit_yolo11n.pt")   # 공통 탐지기 박스를 쓰면 detector 생략

@app.post("/analyze")
async def analyze(...):
    data = await image.read()
    q = quality_model.predict(data)
    pest = pest_model.predict(data, boxes=None)   # 공통 탐지기 박스가 있으면 boxes=[(x1,y1,x2,y2), ...]
    ...
    return AnalyzeOut(..., pest=pest.to_dict())
```

- 모델은 서버를 시작할 때 한 번만 불러온다(불러오기 약 2초).
- CPU에서 과실 하나에 약 0.27초가 걸린다.
- `boxes`는 **원본 사진의 픽셀 좌표**(EXIF 회전을 반영한 뒤 기준)다. 품질 팀과 같은 탐지기 박스를 쓰려면 이 형식으로 넘긴다.

## 4. 결과 형식

```json
{
  "fruits": [
    {"box": [225.8, 390.4, 296.9, 460.6], "label": "damage", "confidence": 0.676,
     "probs": {"healthy": 0.004, "spot": 0.287, "rot": 0.025, "anthracnose": 0.007, "damage": 0.676},
     "uncertain": false, "guide_id": "damage", "guide_type": "group", "tier": "soon",
     "questions": ["damage_split"], "box_source": "detector"}
  ],
  "summary": "damage", "summary_guide_id": "damage", "summary_tier": "soon",
  "model_version": "pest-exp7D-convnext-2026-11-25",
  "status": "ok", "message": "", "skipped_boxes": 0
}
```

과실을 찾지 못하면 판정하지 않는다.

```json
{"fruits": [], "summary": "no_fruit", "summary_guide_id": "", "summary_tier": "none",
 "status": "no_fruit", "message": "사진에서 과실을 찾지 못했습니다. 봉지를 벗긴 과실 하나가 화면을 크게 채우도록 밝은 곳에서 다시 찍어 주세요.",
 "skipped_boxes": 1, "model_version": "pest-exp7D-convnext-2026-11-25"}
```

| 필드 | 뜻 | 앱에서 쓰는 법 |
|---|---|---|
| `label` | 모델 클래스: `healthy`(정상), `spot`(반점병: 세균구멍병·검은별무늬병), `rot`(부패병: 잿빛무늬병·잿빛곰팡이병·그 밖의 부패), `anthracnose`(탄저병), `damage`(구멍·상처: 충해·노린재·열과·물리적 상처) | 내부 값 |
| `guide_id` + `guide_type` | `entry`면 JSON `entries`, `group`이면 `display_groups`(`rot` 부패병, `spot` 반점병, `damage` 구멍·상처)의 항목 | 결과 화면 문구, 대응, 출하 판단 |
| `tier` | `urgent`(지금 바로) · `soon`(이번 주 안에) · `next_season`(올해는 선별, 내년 예방) · `none` | 색과 순서. 문구는 JSON `common.response_tiers` |
| `questions` | 물어볼 질문 규칙 id (`spot_split`, `damage_split`, `spot_vs_damage`) | JSON `disambiguation`에서 질문과 답별 결과를 읽어 화면에 띄움 |
| `uncertain` | 확신도가 0.5 미만 | `unknown` 항목("판별 불가, 다시 찍기 안내")을 보여 줌 |
| `summary*` | 사진 전체에서 가장 급한 병해 | 첫 화면 한 줄 |
| `status`, `message` | `ok` 또는 `no_fruit`(과실을 찾지 못함) | `no_fruit`면 병해 결과 대신 `message`(다시 찍기 안내)를 보여 줌 |
| `skipped_boxes` | 너무 작거나 단색이라 버린 박스 수 | 로그용 |

## 5. 전처리 (학습과 반드시 같아야 함, 코드에 들어 있음)

1. EXIF 회전 반영, RGB
2. 과실 박스를 가로·세로 10% 넓혀 자르기
3. 짧은 변 288px로 리사이즈한 뒤 가운데 288×288
4. ImageNet 정규화 (mean 0.485/0.456/0.406, std 0.229/0.224/0.225)

앱 업로드는 긴 변 1280px 이상, JPEG 품질 85 이상, 필터 없음으로 맞춘다. 과실 하나가 300px 이상 찍히게 안내한다.

## 6. 과실이 없는 사진 처리

| 장치 | 기준 | 근거 |
|---|---|---|
| 사진 전체 판정 안 함 | 과실 박스가 하나도 없으면 `no_fruit` | 처음에는 탐지 실패 때 사진 전체를 판정해서, 초록 단색 이미지가 충해과 96%로 나왔다 |
| 박스 크기 | 짧은 변이 사진 짧은 변의 10% 이상이면서 64px 이상 | 실제 과실 박스는 하위 1%도 14.7% |
| 단색 크롭 제외 | 128×128 흑백 밝기 표준편차 6 미만이면 버림 | 탐지기가 갈색 단색 이미지에 확신도 0.74로 박스를 냈다. 실제 과실 크롭 4,134개의 최솟값은 9.2 |
| 탐지 확신도 | 0.25 유지 | 0.4로 올리면 실제 사진 60장 중 과실 못 찾음이 1장에서 5장으로 늘고, 단색 사진은 막지 못함 |

**남은 위험**: 잎, 땅, 다른 과일처럼 무늬가 있는 사진은 탐지기가 과실로 잡으면 그대로 판정된다. 품질 팀 탐지기 박스를 쓰거나, 미홍 사진으로 탐지기를 확인·보강해야 한다.

## 7. 성능과 한계 (실험 7 D)

외부 평가 두 가지로 확인했다. 학습에 쓰지 않은 사진이다.

| 항목 | NCPMS (농촌진흥청 공식 과실 사진 89장) | test2 v3 (팀 검수 test 정답지 280개) |
|---|---|---|
| 병명 정답 (5클래스) | 72/89 (81%) | 225/280 (80%) |
| 병명 정답, seed 3개 평균 (범위) | 68.3/89 (77%, 66~72) | 228.3/280 (82%, 225~233) |
| 반점병 | 16/17 | 20/26 |
| 부패병 | 34/42 | 55/81 |
| 탄저병 | 15/19 | 6/7 |
| 구멍·상처 | 7/11 | 30/41 |
| 정상 유지 | (NCPMS에 정상 없음) | 114/125 |
| 이상 검출 (병든 과실을 정상이 아니라고 판정) | 87/89 | 153/155 |

**seed에 따른 흔들림**: 같은 설정을 seed만 바꿔 3번 학습하면 NCPMS 66~72(평균 68.3, 약 77%), test2 v3 225~233(평균 228.3, 약 82%)이 나온다. 배포 가중치(seed 42)의 NCPMS 81%는 그중 높은 쪽이다. **새 사진에서 기대할 병명 정확도는 77% 안팎으로 잡는 것이 안전하다.** NCPMS가 89장뿐이라 몇 장 차이로 점수가 크게 움직인다.

- `uncertain` 기준 0.5: NCPMS 89장 중 77장(87%)을 판정하고 12장은 "판별 불가"로 돌린다. 판정한 것의 정답률은 85.7%다(기준이 없으면 80.9%).
- 실험 4 가중치보다 틀린 답의 확신도가 낮다(평균 0.62, 실험 4는 0.83). 그래서 확신도 기준이 오답을 더 잘 거른다.
- **탄저병**: 팀 검수 test의 탄저병 정답 15개를 다시 보니, 8개는 사진 1장에서 나온 마른 과실 박스 5개처럼 탄저병으로 확정하기 어려웠다(v3에서 제외). 확실한 탄저병 7개 중 6개를 맞혔다. 학습 크롭이 52개뿐이라 표본은 여전히 작다. 부패병과 헷갈려도 둘 다 `urgent`라 대응은 같다.
- 정상 미홍 사진이 없어 **미홍 기준 오경보율은 아직 모른다.** test2의 정상 과실(다른 품종)은 125개 중 114개를 정상으로 유지했다.
- 학습하지 않은 병해충(바이로이드, 난괴 등)도 5클래스 중 하나로 판정될 수 있다. 결과는 항상 "의심"으로 표시하고 농업기술센터 문의 안내를 함께 둔다.
- 후보 모델: D, 지식 손실만, CLIP 미세조정을 seed 3개씩 비교했더니 차이가 seed 흔들림 안에 있었다. 그래서 D를 유지한다(CLIP은 크기도 4배). 가중치를 바꿔도 이 인터페이스는 그대로 둔다.

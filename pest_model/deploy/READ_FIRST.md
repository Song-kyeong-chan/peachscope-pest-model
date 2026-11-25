# 병해충 모델: 연동 전에 꼭 알아야 할 것

모델 버전 `pest-exp7D-convnext-2026-11-25` · 대응방안 JSON v0.6.0 · 팀 포도알 병해충 담당

## 1. 무엇을 하는 모델인가
봉지를 벗긴 복숭아 사진에서 과실마다 병해를 5가지로 판정하고, 대응방안과 대응 등급을 붙여 돌려준다.

| 클래스 | 뜻 | 대응 등급 | 앱 질문 |
|---|---|---|---|
| `healthy` | 정상 | none | |
| `spot` | 반점병 (세균구멍병·검은별무늬병) | soon | `spot_split` |
| `rot` | 부패병 (잿빛무늬병·잿빛곰팡이병·그 밖의 부패) | urgent | |
| `anthracnose` | 탄저병 | urgent | |
| `damage` | 구멍·상처 (충해·노린재·열과·물리적 상처) | soon | `damage_split` |

## 2. 설치 (5분, GitHub 로그인 필요 없음)
```bash
git clone https://github.com/Song-kyeong-chan/peachscope-pest-model.git && cd peachscope-pest-model
mkdir -p pest_model/weights                                          # 가중치 2개 받기
curl -L -o pest_model/weights/pest_convnext_exp7D.pt https://github.com/Song-kyeong-chan/peachscope-pest-model/releases/download/v0.7-exp7D/pest_convnext_exp7D.pt
curl -L -o pest_model/weights/fruit_yolo11n.pt https://github.com/Song-kyeong-chan/peachscope-pest-model/releases/download/v0.7-exp7D/fruit_yolo11n.pt
pip install -r pest_model/deploy/requirements.txt
cd pest_model/deploy && python -m pytest -q tests                   # 7개 통과하면 정상
python -m peach_pest 사진.jpg                                        # 사진 한 장 판정 결과(JSON)
```

## 3. 서버에 붙일 때 지킬 것
- **모델은 서버 시작 때 한 번만 불러온다.** 불러오기에 약 2초, CPU에서 과실 하나 판정에 약 0.27초가 걸린다.
- **전처리를 직접 다시 짜지 않는다.** 박스를 10% 넓혀 자르기, 288px, ImageNet 정규화가 `predict()` 안에 들어 있다. 하나라도 다르면 성능이 떨어진다.
- **`boxes`는 원본 사진 픽셀 좌표 `(x1, y1, x2, y2)`다.** EXIF 회전을 반영한 뒤 기준이다. 팀 탐지기 박스를 쓰려면 이 형식으로 넘기고, 넘기지 않으면 내장 탐지기(YOLO11n)를 쓴다.
- **응답에 `model_version`을 함께 저장한다.** 나중에 가중치가 바뀌어도 어떤 모델이 낸 결과인지 추적할 수 있다.

```python
from peach_pest import ConvNeXtPestModel
pest_model = ConvNeXtPestModel("pest_convnext_exp7D.pt", "peach_pest_guide.json", detector="fruit_yolo11n.pt")
result = pest_model.predict(image_bytes, boxes=None).to_dict()
```

## 4. 결과를 화면에 보여 줄 때
| 필드 | 이렇게 처리 |
|---|---|
| `status: "no_fruit"` | 과실을 못 찾았다. 병해 결과 대신 `message`(다시 찍기 안내)를 보여 준다 |
| `uncertain: true` | 가장 높은 확률이 0.5 미만이다. 병명 대신 "판별 불가, 다시 찍기" 안내(JSON `unknown` 항목)를 보여 준다 |
| `summary`, `summary_tier` | 사진에서 가장 급한 병해. 첫 화면 한 줄과 색(urgent 빨강, soon 주황)에 쓴다 |
| `guide_id`, `guide_type` | JSON의 `entries` 또는 `display_groups`에서 증상·조치·출하 판단 문구를 읽는다 |
| `questions` | JSON `disambiguation`에서 질문과 답별 결과를 읽어 띄운다(세부 병명 좁히기) |

**결과는 항상 "○○ 의심"으로 표시하고, 농업기술센터 문의 안내를 함께 둔다.** 사진 진단은 참고용이다.

## 5. 성능 (학습에 쓰지 않은 사진)
| | NCPMS 공식 사진 89장 (전문가 진단) | 팀 검수 test 288개 |
|---|---|---|
| 병명 정답 | 72 (81%) | 225 (78%) |
| 병든 과실을 이상으로 검출 | 87/89 | 160/163 |
| 정상을 정상으로 | (정상 사진 없음) | 114/125 |

판별 불가 기준 0.5를 적용하면 NCPMS 사진의 87%에 답하고, 답한 것의 정답률은 85.7%다.

## 6. 아직 모르는 것 (한계)
- **미홍 과실 사진으로는 아직 검증하지 못했다.** 특히 정상 미홍을 병으로 잘못 판정하는 비율(오경보율)을 모른다. 미홍 사진이 생기면 꼭 측정해야 한다.
- **탄저병이 약하다.** 팀 검수 test에서 15개 중 6개만 맞혔고, 놓친 것은 대부분 부패병으로 판정했다. 둘 다 urgent라 대응 안내는 같다.
- **학습하지 않은 병해충**(바이로이드, 매미충 알 등)도 5가지 중 하나로 판정될 수 있다.
- **과실이 아닌 사진**: 잎이나 땅을 탐지기가 과실로 잡으면 판정된다. 앱 앞단의 복숭아 판별과 함께 쓰는 것을 전제로 한다.

## 7. 지난 전달본(실험 4)을 이미 넣었다면
입력과 출력 형식은 같다. **가중치 파일과 JSON만 바꾸면 된다.** 다만 클래스 이름이 바뀌었으니, 클래스 이름을 직접 쓰는 코드가 있으면 고친다.

| 실험 4 | 실험 7 D |
|---|---|
| `brown_rot`, `others` | `rot` |
| `insect_damage` | `damage` |
| `healthy`, `spot`, `anthracnose` | 그대로 |

## 8. 앞으로
- 더 나은 모델이 나오면 **새 릴리스로 가중치만 교체**한다. 인터페이스는 유지한다.
- 다음 할 일: seed 반복으로 최종 모델 확정(후보: D, CLIP 미세조정), 탄저병 데이터 보강, 대응방안 전문가 확인, 미홍 오경보율 측정.
- 팀에 필요한 것: 미홍 과실 사진(정상 포함), 과실 탐지기 박스 출력 예시 한 줄.

자세한 내용: `docs/pest-model-handoff.html`(전달서), `pest_model/deploy/README.md`(연동), `docs/pest-model-experiment-log.md`(실험 기록)

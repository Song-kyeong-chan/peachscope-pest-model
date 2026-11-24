# PeachScope 병해충 모델

미홍 봉지재배 복숭아 품질 예측 앱(PeachScope)의 보조 기능인 **과실 병해충 판단 모델**이다. 팀 포도알 병해충 담당.

봉지를 벗긴 과실 사진에서 과실을 찾고(YOLO11n), 과실마다 병해를 분류한다(ConvNeXt-Tiny, 6클래스). 결과에는 대응방안 JSON의 항목과 대응 등급(바로 / 이번 주 / 내년 예방)을 붙여 돌려준다.

| 클래스 | 뜻 |
|---|---|
| `healthy` | 정상 |
| `spot` | 반점병 (세균구멍병·검은별무늬병, 앱에서 질문으로 구분) |
| `brown_rot` | 잿빛무늬병 |
| `anthracnose` | 탄저병 |
| `insect_damage` | 충해과 (나방류) |
| `others` | 그 밖의 부패, 흰가루병 |

## 어디를 보면 되나

| 목적 | 위치 |
|---|---|
| 서버에 붙이기 (추론 패키지) | [`pest_model/deploy/`](pest_model/deploy/README.md) |
| 대응방안 데이터 | [`pest_model/knowledge/peach_pest_guide.json`](pest_model/knowledge/peach_pest_guide.json) |
| 실험 과정과 결과 | [`docs/pest-model-experiment-log.md`](docs/pest-model-experiment-log.md) |
| 총괄 전달서 | [`docs/pest-model-handoff.html`](docs/pest-model-handoff.html) |
| 병해충 정리 | [`docs/peach-major-diseases.md`](docs/peach-major-diseases.md) |

## 외부 평가 (농촌진흥청 NCPMS 공식 과실 사진 91장)

이상 검출 93% · 부패병 그룹 49/61 · 반점병 11/17 · 잿빛무늬병 25/31 · 충해과 3/8. 정상 미홍 과실 오경보율은 아직 측정하지 않았다.

## 저장소에 없는 것

- 모델 가중치(`*.pt`): 용량 때문에 드라이브로 따로 전달한다.
- 사진 데이터: `~/peachscope-data`에 두고 올리지 않는다. NCPMS 사진은 저작권 때문에 평가에만 썼다.
- API 키(`.env`).

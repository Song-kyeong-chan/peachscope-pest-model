# PeachScope 병해충 모델

미홍 봉지재배 복숭아 품질 예측 앱(PeachScope)의 보조 기능인 **과실 병해충 판단 모델**이다. 팀 포도알 병해충 담당.

봉지를 벗긴 과실 사진에서 과실을 찾고(YOLO11n), 과실마다 병해를 분류한다(ConvNeXt-Tiny, 5클래스, 실험 7 D). 결과에는 대응방안 JSON의 항목, 대응 등급(바로 / 이번 주 / 내년 예방), 앱에서 물어볼 질문을 붙여 돌려준다.

| 클래스 | 뜻 | 앱 질문 |
|---|---|---|
| `healthy` | 정상 | |
| `spot` | 반점병 (세균구멍병·검은별무늬병) | `spot_split` |
| `rot` | 부패병 (잿빛무늬병·잿빛곰팡이병·그 밖의 부패) | |
| `anthracnose` | 탄저병 | |
| `damage` | 구멍·상처 (충해·노린재·열과·물리적 상처) | `damage_split` |

## 어디를 보면 되나

| 목적 | 위치 |
|---|---|
| **연동 전에 꼭 알아야 할 것** | [`pest_model/deploy/READ_FIRST.md`](pest_model/deploy/READ_FIRST.md) |
| 총괄 전달서 | [`docs/pest-model-handoff.html`](docs/pest-model-handoff.html) |
| 서버에 붙이기 (추론 패키지) | [`pest_model/deploy/`](pest_model/deploy/README.md) |
| 대응방안 데이터 (v0.6.0) | [`pest_model/knowledge/peach_pest_guide.json`](pest_model/knowledge/peach_pest_guide.json) |
| 실험 과정과 결과 | [`docs/pest-model-experiment-log.md`](docs/pest-model-experiment-log.md) |
| 학습·평가 코드 | `pest_model/build_exp7.py`(라벨), `train_exp7.py`(학습), `rescore_test2.py`(test 재채점), `pick_threshold.py`(판별 불가 기준) |
| 병해충 정리 | [`docs/peach-major-diseases.md`](docs/peach-major-diseases.md) |

## 외부 평가 (실험 7 D)

| | NCPMS 공식 과실 사진 89장 (전문가 진단) | 팀 검수 test 288개 |
|---|---|---|
| 병명 정답 | 72/89 (81%) | 225/288 (78%) |
| 이상 검출 | 87/89 | 160/163 |
| 정상 유지 | (정상 사진 없음) | 114/125 |

확신도 0.5 미만은 "판별 불가"로 돌린다(NCPMS 87% 판정, 판정한 것의 정답률 85.7%). 미홍 정상 과실 오경보율은 아직 측정하지 않았다.

## 저장소에 없는 것

- 모델 가중치(`pest_convnext_exp7D.pt`, `fruit_yolo11n.pt`): 용량 때문에 커밋하지 않고 [Releases `v0.7-exp7D`](https://github.com/Song-kyeong-chan/peachscope-pest-model/releases/tag/v0.7-exp7D)에 올렸다. 받아서 `pest_model/weights/`에 둔다.
  `gh release download v0.7-exp7D -R Song-kyeong-chan/peachscope-pest-model -p '*.pt' -D pest_model/weights`
- 사진 데이터: `~/peachscope-data`에 두고 올리지 않는다. NCPMS 사진은 저작권 때문에 평가에만 썼다.
- API 키(`.env`).

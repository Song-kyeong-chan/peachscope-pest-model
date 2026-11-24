# pest_model — 복숭아 과실 병해 분류 베이스라인

공개 데이터(Mendeley PDDICP, CC BY 4.0)로 과실 3클래스(정상 / 세균구멍병 / 잿빛무늬병)를 분류한다.
실험 과정과 결과는 [`docs/pest-model-experiment-log.md`](../docs/pest-model-experiment-log.md)에 기록한다.

## 실행

```bash
bash pest_model/run_all.sh            # 전체 (다운로드 2GB + 학습 + 5-fold CV, 약 50분, Apple M4 기준)
SKIP_CV=1 bash pest_model/run_all.sh  # 교차검증 빼고 (약 15분)
```

- 처음 실행하면 `pest_model/.venv`를 만들고 torch, timm, scikit-learn을 설치한다.
- 사진은 `~/peachscope-data`(`DATA_ROOT`로 변경 가능)에 저장한다. 저장소에는 올리지 않는다.
- 이미 끝난 단계(가상환경, 다운로드, 분할)는 건너뛴다.

## 단계

| 단계 | 파일 | 출력 |
|---|---|---|
| 준비 | `prepare.py` | `~/peachscope-data/pddicp3/images/`, `splits.csv` (8:1:1, seed 42) |
| 편향 점검 | `bias_check.py` | `results/bias_check.json` |
| 학습·비교 (1회 분할) | `train.py` | `results/results.json`, `results/*.pt` (git 제외) |
| 결과표 | `report.py` | `results/results.md` |
| Grad-CAM | `gradcam.py` | `results/gradcam_resnet50.jpg` |
| 5-fold 교차검증 | `cv.py` | `results/cv/cv_results.json` |

## 주의

- 학습된 가중치(`*.pt`)는 git에 올리지 않는다. 공유가 필요하면 비공개 저장소를 쓴다.
- PDDICP의 `bacterial spot` 폴더는 영양결핍 잎 사진의 복사본(라벨 오류)이라 쓰지 않는다.
- 정상은 원거리, 병해는 근접 사진이라 촬영 거리 편향이 있다. 점수를 해석할 때 `bias_check` 기준선과 같이 본다.

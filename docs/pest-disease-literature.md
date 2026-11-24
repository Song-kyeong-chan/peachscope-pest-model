# 복숭아 병해충 탐지 모델 문헌조사

> 웹 검색 결과의 초록·요약을 기준으로 정리했다(2026-11-09). 인용 전에 원문에서 수치와 저자를 재확인할 것.
> 관련 문서: [중간발표 피드백 반영](midterm-feedback.md) · [품질·수확적기 문헌조사](peach-literature-review.md)

---

## 0. 요약

- **가장 가까운 선행연구 3편**
  - Farman et al. 2022: 스마트폰으로 현장 촬영, 과실·잎 6클래스, 데이터 공개.
  - Yao et al. 2021·2022: 화중농대, 복숭아 병해 7종을 분류하고 분할.
  - Masuda et al. 2023: RGB 사진 한 장으로 당도·경도·**복숭아심식나방 피해**를 동시에 진단.
- **바로 받을 수 있는 공개 데이터**: Mendeley PDDICP(약 2,500장, CC BY 4.0). Roboflow에도 소규모 복숭아 병해 bbox 데이터가 있다. AI Hub에는 복숭아 병해 데이터가 없다.
- **국내에서 실제로 피해가 큰 병해충**(유기 과원 6곳 조사, 출하 불가 과실 비율): 복숭아순나방 13.5%, 잿빛무늬병 13.0%, 복숭아심식나방 7.3%, 세균구멍병 7.3%. 클래스 우선순위를 정하는 근거로 쓴다.
- **공백**: 해충 **피해과**를 영상으로 탐지한 딥러닝 연구는 거의 없다(Masuda 2023의 심식나방 피해 진단 정도). 해충 연구는 대부분 페로몬 트랩 영상의 성충 계수다. 이 공백이 우리 과제의 차별점이 될 수 있다.

---

## 1. 복숭아 병해 영상 인식 논문

| 연구 | 대상 | 데이터 | 방법 | 핵심 결과 |
|---|---|---|---|---|
| **Farman et al., *CMC* 71(1) 2022** ([논문](https://techscience.com/cmc/v71n1/45366/html), [데이터](https://data.mendeley.com/datasets/3pmj85snvw/1)) | 과실+잎 6클래스: 정상, 잿빛무늬병, 수지병, 영양결핍, 세균구멍병(잎), 세균구멍병(과실) | **약 2,500장, 클래스당 약 400장**. 파키스탄 과원에서 **스마트폰**(Galaxy S8 등 4종)으로 다양한 조명에서 촬영. **CC BY 4.0 공개** | EfficientNet B0~B5, ResNet50, InceptionV3, MobileNetV2, AlexNet 비교. 전체 미세조정 | EfficientNet-B2가 최고: 정확도 96.6%, 민감도·정밀도 90%, 특이도 98%. B3~B5는 과적합. SSD로 과실을 잘라내는 단계는 데이터 준비용으로만 쓰고, 탐지+분류 통합은 향후 과제로 남김 |
| **Yao et al., *Plant Methods* 17:36, 2021** — L2MXception ([DOI](https://doi.org/10.1186/s13007-021-00736-3)) | 7개 병해: 잿빛무늬병, 탄저병, 검은별무늬병, 세균구멍병, 수지병, 흰가루병, 잎오갈병 | PDID 1,560장 (아래 1-1 참고). 분할 1,251 / 156 / 153 (약 8:1:1) | 주요 CNN 7종 비교 + **L2M Loss**(L2 정규화와 평균 항을 더한 손실) | Xception+L2M **검증** 정확도 93.85%. "28.48% 향상"은 원본 Xception의 65.37%와 비교한 값이다. 정규화 없이 최고였던 모델은 DenseNet169(89.32%) |
| **Yao et al., *Front. Plant Sci.* 2022** ([링크](https://www.frontiersin.org/articles/10.3389/fpls.2022.876357)) | 위 7개 병해, **과실 위주** | 1,560장(과원·실내 촬영, 병원균 분리로 진단 확인) → 초·중·말기로 나눠 21클래스, 증강 후 5,627장 | Mask R-CNN / Mask Scoring R-CNN + Focal Loss | bbox mAP50 최고 0.771, 분할 mAP50 최고 0.463 |
| **Masuda et al., *Postharvest Biol. Technol.* 2023** ([요약](https://bohrium.dp.tech/paper/arxiv/39372855), DOI 10.1016/j.postharvbio.2023.112348) | 과실 7개 형질: 과피색, **경도, 당도**, 조기연화, 핵할, 수침(밀병), **복숭아심식나방 피해** | 일본(오카야마대, Akagi 연구실) | **RGB 사진만으로** CNN 이진분류 + 회귀. 특징 시각화로 증상 부위 설명 | 여러 형질을 RGB로 동시에 진단 가능. **우리 앱(품질 + 병해충)과 구조가 가장 비슷하다.** 형질별 정확도는 원문 확인 필요 |
| AgriEngineering 2026 (Masih et al., [PDF](https://mdpi-res.com/d_attachment/agriengineering/agriengineering-08-00368/article_deploy/agriengineering-08-00368.pdf)) | 잿빛무늬병, 잎오갈병 | 열대 고지대 현장 촬영 | CNN + 전처리 ablation + 교차검증, 웹앱(DurAPP) | 저가 진단 도구 목표. **"전처리 효과를 ablation으로 검증"하는 구성이 피드백 1·3번과 맞다.** 수치는 PDF 본문 확인 필요 |
| *Ecological Informatics* 2021 ([X-MOL](https://www.x-mol.com/paper/1358610269332086784)) | 잎 세균병 | 잎 이미지 | CNN | 정확도 98.75%, 이미지당 0.185초 |
| DASA 2020 ([PDF](https://www.di.ubi.pt/~hugomcp/doc/DASA_2020.pdf)) | 과실 4클래스 | 313장 (공개 출처) | MobileNetV2 전이학습 | 작은 데이터에서의 전이학습 사례 |
| *Scientific Reports* 2025 ([PDF](https://rcastoragev2.blob.core.windows.net/6f713250f40f5cfe641129da9ded15a6/41598_2025_Article_92143.PMC11889226.pdf)) | 잎 | PlantVillage | YOLOv3 vs YOLOv4 | YOLO 비교 틀 |
| *Electronics* 2024 ([MDPI](https://www.mdpi.com/2079-9292/13/24/4973)) | 잎오갈병 | 자체 수집 | YOLOv8 개선 | 원본 YOLOv8 대비 향상 |

### 1-1. Yao et al. 2021 (L2MXception)의 데이터 수집 방법 ([PMC 전문](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8017885/))

- **누가**: 화중농업대학 식물과학기술학원 Luo Chaoxi 교수팀. 중국 국가복숭아산업기술체계(National Peach Industry Technology System)의 **과수 병해 방제 전문가 팀**이다. 사진 제공과 라벨링을 모두 Luo 교수가 맡았다.
- **어떻게 모았나**: 두 가지 경로를 썼다.
  1. 연구실이 이미 갖고 있던 사진과, 복숭아 산업기술체계 안의 다른 전문가들에게 협력으로 받은 사진
  2. 실내와 과원에서 직접 대량 촬영
- **무엇을 찍었나**: 병마다 주로 감염되는 부위를 찍었다. 과실에 주로 생기는 병은 과실 사진 위주로 모았다.
- **라벨 정확도 확보 방법**: 증상이 비슷해 헷갈리는 병은 병원균을 조직이나 단포자로 분리하고, 형태를 관찰하고, 분자생물학적으로 동정해서 확정했다.
- **7개 클래스의 정의** (논문 원문 번역): ① 과실·잎의 갈색부패균 ② 과실·잎의 탄저병균 ③ **과실·가지·잎의 검은별무늬병균(scab)** ④ 과실·가지·잎의 세균구멍병균 ⑤ 가지의 수지병균 ⑥ 과실·잎의 흰가루병균 ⑦ 잎의 잎오갈병균
- **전처리**: 224×224 또는 299×299로 리사이즈하고 Z-score로 정규화했다. 회전 ±10°, 이동 15%, 크기 0.9~1.1배, 기울임 10° 증강을 썼다.
- **공개 여부**: "reasonable request" 시 교신저자가 제공한다고 되어 있다. 교신저자는 Ni Fuchuan(화중농대 정보학원)이다.
- **방법상 한계** (우리 검증 설계에 반영할 것)
  - 증강을 **검증·테스트 데이터에도** 적용했다.
  - 핵심 수치 93.85%는 테스트가 아니라 **검증** 정확도다.
  - 같은 병반을 여러 장 찍은 사진이 분할 사이에 섞였는지는 언급이 없다.

**병해 이름 대조** (논문마다 같은 병을 다르게 부르니 라벨 이름을 통일할 때 참고)

| 영문 | 국문 | Yao 2022 원본 장수 |
|---|---|---|
| Scab | 검은별무늬병 | 654 |
| Bacterial shot hole (= bacterial spot) | 세균구멍병 | 427 |
| Anthracnose | 탄저병 | 157 |
| Brown rot (*Monilinia*) | 잿빛무늬병 | 94 |
| Gummosis | 수지병 | 91 |
| Leaf curl | 잎오갈병 | 87 |
| Powdery mildew / rusty spot (*Podosphaera pannosa*) | 흰가루병 / 녹얼룩점 | 50 |

## 2. 해충 관련 연구

| 연구 | 내용 | 시사점 |
|---|---|---|
| Masuda et al. 2023 (위) | RGB로 **복숭아심식나방 피해과** 진단 | 피해과 영상 진단의 거의 유일한 딥러닝 선례 |
| Smart trap + ResNet18 ([링크](https://www.bohrium.com/paper-details/deep-learning-embedded-into-smart-traps-for-fruit-insect-pests-detection/864986920172650558-2616)) | 지중해과실파리·**복숭아순나방** 트랩 영상 분류, 모델 경량화 | 해충 연구는 대부분 트랩 성충 계수 |
| Ding & Taylor 2016 ([arXiv](https://arxiv.org/pdf/1602.07383)) | 트랩 영상 나방 탐지·계수 (코들링나방) | 저화질 영상 문제 |
| 페로몬 전격 트랩 머신비전 2025 ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11922902/)) | 끈끈이판 해충 탐지 | 사체 훼손으로 오탐·미탐 발생 |
| Zhao et al. 2016 (영상처리, 비딥러닝) | 복숭아순나방 트랩 영상 자동 계수, 피해과율 5% 도달 시점과 연결 | 트랩 계수 → 방제 적기 연결 사례 |

**판단**: 앱은 과실을 찍으므로 트랩 방식은 우리 과제와 맞지 않는다. **피해과 판정(구멍, 진물, 똥)** 쪽으로 간다.

## 3. 과실 탐지 논문 (1단계 탐지기 설계용)

| 연구 | 방법 | 결과 | 참고할 점 |
|---|---|---|---|
| EMA-YOLO, *Sensors* 2024 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11207478)) | YOLOv8 + EMA 어텐션, 미숙 황도 | — | 복잡한 배경, 가림 |
| PdYOLO 2024 ([링크](https://library.kab.ac.ug/Record/doaj-art-1227e84c19284e379fc1e929c710145d/Description)) | YOLOv8s 경량화 | — | 서버 부담 감소 |
| YOLO-Peach, *Agronomy* 2024 ([MDPI](https://mdpi.com/2073-4395/14/8/1628)) | YOLOv8s 경량 | mAP50 0.993 | 어린 과실 계수 |

## 4. 국내 연구

| 연구 | 내용 | 활용 |
|---|---|---|
| **김민기 외, 한국유기농업학회지 25(3) 2017** ([링크](https://koreascience.kr/article/JAKO201725650934533.page)) | 유기 복숭아 과원 6곳, 2015~2017년. 충해 31.6%, 병해 24.1%. **복숭아순나방 13.5%, 잿빛무늬병 13.0%, 복숭아심식나방 7.3%, 세균구멍병 7.3%** | **클래스 우선순위 근거.** 관행(약제) 재배에서는 비율이 다를 수 있음 |
| Shin et al., 한국균학회지 46(2) 2018 ([링크](https://koreascience.kr/article/JAKO201817968657849.page)) | 국내 복숭아 과실의 흰가루·녹얼룩점 증상 모두 *Podosphaera pannosa*가 원인 | 두 증상을 한 클래스로 묶을 근거 |
| '미황' 성숙도 분류, 생물환경조절학회지 2022 ([링크](https://koreascience.kr/article/JAKO202232055723394.page?lang=ko)) | VGG16, InceptionV3로 미숙·성숙·과숙 분류. 실내·야외 각 2,800장 | 국내 복숭아 RGB 딥러닝 선례 |
| '미황' 성숙도 자동 분류, 원예과학기술지 2024 ([PDF](https://www.hst-j.org/articles/pdf/1Qry/kshs-2024-042-01-7.pdf), [KCI](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003052284)) | EfficientNet, YOLOv5, ViT 비교, 로봇 수확용. 과정부 a* 기준 정확도 94.7~98.2% | **알고리즘 3개 비교 구성의 좋은 예**(피드백 4번). 국립원예특작과학원 공저라 데이터 문의 후보 |

국내에서 복숭아 **병해**를 딥러닝으로 분류한 논문은 이번 검색에서 찾지 못했다.

## 5. 공개 데이터셋

| 데이터 | 복숭아 병해? | 내용 | 비고 |
|---|---|---|---|
| **Mendeley PDDICP** ([링크](https://data.mendeley.com/datasets/3pmj85snvw/1)) | ✅ | **실제 공개분은 1,108장**(논문의 약 2,500장보다 적음), 폴더별 분류용, 라벨 없음 | **CC BY 4.0.** 과실로 쓸 수 있는 건 정상·잿빛무늬병·세균구멍병 3클래스뿐. 아래 5-1 점검 결과 참고 |
| **Roboflow "peach detection"** (Islamia College Peshawar) ([링크](https://universe.roboflow.com/islamia-college-peshawar-x56lw/peach-detection-rfv6n)) | ✅ | **3,914장이지만 실제 원본은 약 1,488묶음**(원본 1장당 증강본 2~3장, 2026-11-14 확인), 3클래스(brown_rot, healty_Peach, shot_hole), bbox, 분할 70/20/10. 공개 모델 mAP50 94.3%는 **증강본이 분할에 흩어진 누수 때문에 과대평가**됐다 | **CC BY 4.0.** 과실 현장 사진 위주. 파일명이 `GOOD-`, `bad-`, 날짜형 등으로 섞여 있어 여러 출처를 합친 것으로 보임. 황도·녹색 과실도 섞여 있음. PDDICP와 같은 기관이라 **이미지 중복 확인 필요** |
| Roboflow "Peach Disease" (zeerox) ([링크](https://universe.roboflow.com/zeerox/peach-disease)) | ✅ | **원본 1,081장**(v8은 증강 포함 2,811장, 분할 92/4/4), 5클래스(bacterial_spot, brown_rot, healthy_peach, shot_hole, shot_hole_leaf) | **CC BY 4.0. 사진은 PDDICP와 같다**(아래 5-4). PDDICP에 과실 단위 라벨을 붙인 버전 |
| Roboflow "PEST DETECTION" (YooSeop Jang) ([링크](https://universe.roboflow.com/yooseop-jang-a4fo7/pest-detection-jzeof)) | 해충 | **116장**, 인스턴스 분할, 3클래스(사과애모무늬잎말이나방, 복숭아순나방, 복숭아심식나방). mAP50 69.7% | CC BY 4.0. 성충 사진이고 양이 적어 **피해과 탐지에는 쓰기 어려움** |
| Mendeley AgriVision-Apricot (Bilal Jan) ([링크](https://data.mendeley.com/datasets/5wk5wpzczw/1)) | 근연종 | 원본 806장, 9클래스 (증강본 1,350장은 별도) | CC BY 4.0. 아래 5-3 점검 결과 참고 |
| Mendeley "Plum Leaf and Fruit" (Data in Brief 2025) ([링크](https://data.mendeley.com/datasets/w7sdx55m7z/1)) | ❌ | 3,554장, 6클래스, 흰 배경 누끼 사진 | **Prunus 자두가 아니라 인도대추(*Ziziphus*)로 보임.** 아래 5-3 참고 |
| ~~Mendeley bsr2vzhrzr~~ | ❌ | 처음에 자두 데이터로 잘못 기재했던 항목. 실제로는 포도·땅콩·구아바·콩 데이터 | 사용 안 함 |
| PlantDoc (Kaggle) | 잎 일부 | 13종 28클래스 중 복숭아 잎 | 과실이 아니라 우선순위 낮음 |
| AI Hub 노지 작물 질병 ([147](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=147)) | ❌ | 10작물 20병해, bbox + 피해 단계(0~3) | 라벨 구조만 참고 |
| AI Hub 노지 작물 해충 ([148](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=148)) | △ | 복숭아혹진딧물 21,242장 | 어떤 작물에서 찍었는지 미확인 |
| AI Hub 과수화상병 ([146](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=146)) | ❌ | 사과·배, 사과탄저병 21,452건 | 탄저병 사전학습용 |
| AI Hub 농산물 품질(QC) ([149](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=149)) | ❌ | 10작물 등급, 다각도 촬영 | 촬영 프로토콜 참고 |

AI Hub는 내국인만 신청할 수 있고, 승인 후 다운로드한다. 외부 데이터는 모두 라이선스를 확인한 뒤 쓴다.

**AI Hub '복숭아' 검색 결과 (2026-11-09)**: 4건이 나왔지만 복숭아를 주제로 한 데이터는 없다(148 해충, 476 소형 객체, 242 음식, 71887 농작업 로봇). 병든 복숭아 데이터는 없다.

**AI Hub 71887 단계적 사고 기반 농작업 로봇 데이터** ([링크](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71887))
- 2025년 구축, 2026-06 개방. 주관은 유클리드소프트, **수집은 경북대 산학협력단**.
- 사과·배·복숭아·자두의 꽃 솎기, 열매 솎기, 가지치기, 수확 작업 영상에서 뽑은 RGB와 깊이 이미지(OAK-D 카메라, 1920×1080), 장갑 센서, 환경 센서(온도·습도·조도). 라벨은 bbox + 질의응답 + CoT(JSON).
- **복숭아 라벨 4,960건**: 열매 솎기 600, 가지치기 1,348, 수확 3,012. 수확/비수확 구분은 배에만 있고 복숭아에는 없다.
- 대상 객체마다 bbox, 크기, 유형, 상태, 위치, **HSV 값**이 붙어 있다.
- **병해 라벨이 없고 품종 필드도 없다.** 병해충 모델에는 쓸 수 없다.
- 쓸 수 있을 만한 곳: ① 국내 과원 복숭아 사진으로 1단계 과실 탐지기를 사전학습 ② 수확 작업 사진을 수확기 과실의 예시로 활용 (피드백 10번). 실제 사진의 품질과 봉지 착용 여부는 샘플을 받아 확인해야 한다.

### 5-1. Mendeley PDDICP 점검 결과 (2026-11-09 다운로드)

- 파일: `Peach_Dataset.zip`, 2.08 GB. SHA-256이 Mendeley에 공시된 값과 일치했다.
- 촬영: 기기는 HUAWEI LUA-U22 992장, OPPO F5 80장, EXIF 없음 36장(PNG). 해상도는 대부분 4.9 MP.

| 폴더 | 장수 | 실제 내용 | 사용 여부 |
|---|---|---|---|
| healthy | 353 | 나무에 달린 **과실 여러 개**를 멀리서 찍은 사진 | ✅ 정상 과실 (과실 하나를 크게 찍은 사진이 아님) |
| shot hole | 540 | **과실 근접 사진**(손에 들고 찍은 것 다수). 세균구멍병 반점 | ✅ 가장 쓸 만함 |
| brown rot | 49 | 나무에 달린 썩은 과실(미라과 포함) | ✅ 다만 양이 적음 |
| shot hole leaf | 36 | 잎 | ❌ 잎 |
| nutreint deficency | 75 | 잎 황화 | ❌ 잎 |
| **bacterial spot** | 55 | **55장 전부 `nutreint deficency` 폴더의 사진과 파일이 똑같음**(파일명도 `nutrient dificiency (n).jpg`) | ❌ **라벨 오류**. 쓰면 안 됨 |

- 완전 중복: 55그룹, 남는 파일 64장. 그중 55그룹은 위 bacterial spot ↔ 영양결핍 사이에 걸쳐 있다.
- 유사 중복: dHash 거리 4 이하인 쌍이 1개 있다(`shot hole (72).jpg` ↔ `shot hole on leaf (2).png`). 과실 사진이 잎 폴더에 들어간 것으로 보인다.
- 논문(Farman 2022)은 수지병 클래스와 약 2,500장을 보고했지만, 공개본에는 수지병이 없고 장수도 적다. **논문 수치를 이 데이터의 성능 기준으로 쓰면 안 된다.**
- **결론**: 과실 3클래스로 실제 쓸 수 있는 장수는 약 940장(정상 353 + 세균구멍병 540 + 잿빛무늬병 49)이다.
  - 정상은 원거리, 병해는 근접 사진이라 **촬영 거리가 클래스와 겹친다.** 그대로 분류기를 학습하면 모델이 병반이 아니라 "가까이 찍었는가"를 배울 위험이 있다.
  - 1단계 탐지기로 과실을 잘라낸 뒤 2단계 분류에 쓰거나, bbox를 다시 라벨해서 탐지용으로 써야 한다.
  - Roboflow Islamia(3,914장)와 같은 기관 자료이므로 합칠 때 중복을 제거해야 한다.

---

### 5-3. 근연종(살구·자두) 데이터 점검 결과 (2026-11-09)

**AgriVision-Apricot**: 원본 806장. SHA-256 해시가 전부 일치했다.

| 클래스 | 장수 | 실제 내용 |
|---|---|---|
| shot_hole_fruit | 174 | 나무에 달린 **과실 여러 개를 멀리서** 찍은 사진. 병반이 작게 보임 |
| healthy | 158 | 나무·가지 단위 원거리 사진 |
| shot_hole_leaves | 149 | 잎 |
| gummosis_and_galls | 132 | 줄기·가지(수지병, 혹) |
| other_disorder (water stress) | 74 | 시든 잎과 과실 |
| brown_rot | 47 | 가지에 말라붙은 꽃·잎·과실(미라과). 과실 근접 사진은 적음 |
| wilting | 37 | 나무 전체 |
| dieback | 21 | 가지 고사 |
| powdery_mildew | 14 | 잎 |

- 촬영 기기가 PDDICP 복숭아와 같다(HUAWEI LUA-U22, OPPO F5). 같은 파키스탄 연구 그룹(Bilal Jan, Islamia College)의 데이터다. 두 데이터 사이에 중복 이미지는 없었다(md5와 dHash로 확인).
- **한계**: 대부분 나무·가지 단위 원거리 사진이다. 과실 병반을 크게 찍은 사진이 거의 없다. 흰가루병(14)과 잿빛무늬병(47)은 장수도 너무 적다.
- **용도**: 세균구멍병 과실·잎(323장)은 보조 데이터로 쓸 만하다. 단, 1단계 탐지로 과실을 잘라낸 뒤에 써야 한다. 수지병(132장)은 가지 사진이라 과실 앱 범위 밖이다.

**Plum Leaf and Fruit (Data in Brief 2025, 방글라데시)**: Processed 버전 3,554장(512px 안팎, 배경 제거)을 확인했다.
- 클래스: Bacterial Spot 592, Shot Hole 643, Healthy Leaf 735, Wilted Leaf 548, Healthy Fruit 560, Unhealthy Fruit 476.
- **사진을 보면 Prunus 자두가 아니라 인도대추(*Ziziphus mauritiana*, 현지 이름 boroi/kul)로 보인다.** 근거는 셋이다. ① 잎맥 3개가 잎자루에서 갈라짐 ② 작고 둥근 연녹색 과실 ③ 수집 시기가 12~2월로 대추 수확기와 같음.
- 그래서 "Shot Hole"도 세균구멍병이 아니라 벌레 구멍일 가능성이 높다. 흰 배경 실험실 사진이라 현장 사진과도 다르다. **복숭아 전이학습용으로는 쓰지 않는다.**

### 5-4. Roboflow zeerox "Peach Disease" 점검 결과 (2026-11-13)

로그인 없이 볼 수 있는 50장과 공개 메타데이터로 확인했다. 전체 다운로드에는 로그인이 필요하다.

- **사진 출처 = PDDICP**: 파일명이 같은 패턴이다(`shot hole (351).jpg`, `healthy (182).jpg`, `brown rot (11).jpg`, `shot hole on leaf (13).png`). 썸네일 4장을 받아 PDDICP 원본과 비교했더니 dHash 거리가 256비트 중 1~4로, 사실상 같은 사진이다. 파일명이 UUID인 잎 사진처럼 다른 출처 사진도 일부 섞여 있다.
- **이 데이터의 가치는 라벨이다**: PDDICP는 폴더 분류뿐이지만, zeerox에는 객체 단위 라벨이 있다.
  - healthy_peach: 나무에 달린 **과실 하나하나**를 bbox 또는 다각형으로 표시 → **1단계 과실 탐지기 학습에 바로 쓸 수 있다**
  - shot_hole: **과실 전체**를 다각형 하나로 표시(병반 단위 아님) → 과실 단위 분류용
  - shot_hole_leaf, bacterial_spot: 잎 다각형
  - bbox와 다각형이 섞여 있다. YOLO 탐지 형식으로 내보내면 다각형도 bbox로 바뀌므로 문제없다.
- **버전 v8 주의**: train 2,595장은 원본 약 865장 × 3배 증강(뒤집기, ±10° 회전, 블러, 노이즈)이다. valid 107장, test 109장은 증강 전 원본이다. 내보낸 train에 같은 사진의 변형이 3장씩 들어 있으므로 **우리 방식(8:1:1, 과실 단위)으로 다시 나누려면 증강 없는 원본이 필요하다**. 원본은 Fork Dataset으로 가져와 증강 없이 새 버전을 만들면 된다.
- **PDDICP와 합치면 안 된다**: 같은 사진이므로 둘을 합쳐 쓰면 train/test 누수가 생긴다. **PDDICP 대신 zeerox 라벨본 하나만 쓴다.**
- PDDICP의 bacterial spot 폴더 오류(영양결핍 잎 사진 55장의 복사본)가 zeerox의 bacterial_spot 클래스에 어떻게 반영됐는지는 전체를 받아서 확인해야 한다.

### 5-2. 주요 병해별 데이터 확보 현황

| 병해 | 감염 부위 | 바로 받을 수 있는 공개 데이터 | 요청해야 받을 수 있는 데이터 |
|---|---|---|---|
| 갈색부패병(잿빛무늬병, brown rot) | 과실·잎 | PDDICP 과실 49장, Roboflow Islamia·zeerox의 brown_rot, Mendeley 살구 | Yao 2022 (94장) |
| 세균구멍병(shot hole) | 과실·가지·잎 | PDDICP 과실 540장·잎 36장, Roboflow 2종, PlantVillage 잎(bacterial spot), 살구 과실 174장·잎 149장 | Yao 2022 (427장) |
| 탄저병 | 과실·잎 | **없음** (AI Hub 사과탄저병 21,452건은 다른 작물) | Yao 2022 (157장) |
| 흰가루병 | 과실·잎 | **없음** | Yao 2022 (50장) |
| 겹무늬병 → 원문은 **scab(검은별무늬병)** | 과실·가지·잎 | **없음** | Yao 2022 scab (654장) |
| 수지병(gummosis) | 가지 | **없음** (Farman 논문에는 있으나 공개본에서 빠짐) | Yao 2022 (91장) |
| 잎오갈병(잎말림병, leaf curl) | 잎 | **없음** | Yao 2022 (87장), Electronics 2024 자체 데이터 |

- Yao 2022의 데이터 공개 문구: "raw data … will be made available by the authors, without undue reservation". 저자에게 메일로 요청하면 받을 수 있다는 뜻이다.
- **7종을 모두 갖춘 데이터는 Yao(화중농대) 데이터뿐이다.** 1순위 확보 대상이다.

## 6. 모델 설계 방향 (초안)

**앱이 과실을 찍으므로 과실 표면 병해와 해충 피해과를 대상으로 한다.**

1. **1단계 과실 탐지**: YOLO(v8/v11)로 복숭아 bbox를 찾는다. 피드백 7번의 탐지 모델과 공유한다.
2. **2단계 병해충 판정**: 잘라낸 과실을 분류한다. Farman 2022와 같은 구조이고, 그 논문이 향후 과제로 남긴 "탐지+분류 통합"을 우리가 한다.
3. **후보 클래스** (국내 피해율 순): 정상 / 잿빛무늬병 / 세균구멍병 / 해충 피해과(복숭아순나방·심식나방) / 탄저병 / 검은별무늬병. 데이터가 적으면 "정상 vs 병해 vs 해충 피해" 3클래스로 시작한다.
4. **학습 전략**: PDDICP와 Roboflow 데이터로 사전학습 → 우리가 직접 찍은 미홍 사진으로 미세조정 → 우리 사진으로만 test.
5. **검증** (피드백 3·4번): 8:1:1, 과실 단위 분할. 모델 3~4개를 비교한다.
   - 분류: EfficientNet-B2 / MobileNetV3 / ResNet50 / ViT
   - 탐지: YOLOv8 / YOLOv11 / RT-DETR
   - 지표: 정확도, 클래스별 F1, 혼동행렬, mAP50
6. **설명**: Grad-CAM으로 모델이 병반을 보고 있는지 보여준다(Masuda 2023처럼).

**예상 난점**
- 데이터 불균형 → Focal Loss 또는 L2M Loss, 클래스 가중치.
- 봉지 재배 미홍은 병해가 적을 수 있다 → 수집 가능한 장수를 먼저 확인한다. 부족하면 근연종(살구) 데이터를 보조로 쓴다.
- 해외 데이터와 국내 미홍은 품종과 촬영 환경이 다르다 → 피드백 1번의 보정 보드로 촬영하고, 국내 사진으로 test한다.

## 7. 참고문헌 목록 (조사한 논문 전체)

### 복숭아 병해 인식
1. Farman, H., Ahmad, J., Jan, B., Shahzad, Y., Abdullah, M., Ullah, A. (2022). EfficientNet-Based Robust Recognition of Peach Plant Diseases in Field Images. *Computers, Materials & Continua*, 71(1), 2073–2089. [논문](https://techscience.com/cmc/v71n1/45366/html) · [데이터셋(Mendeley)](https://data.mendeley.com/datasets/3pmj85snvw/1)
2. Yao, N. et al. (2021). L2MXception: an improved Xception network for classification of peach diseases. *Plant Methods*, 17, 36. [DOI](https://doi.org/10.1186/s13007-021-00736-3) · [DOAJ](https://doaj.org/article/0b2833e3e2a348c9946e14de07ddb35b)
3. Yao, N., Ni, F., Wu, M., Wang, H., Li, G., Sung, W.-K. (2022). Deep Learning-Based Segmentation of Peach Diseases Using Convolutional Neural Network. *Frontiers in Plant Science*, 13, 876357. [링크](https://www.frontiersin.org/articles/10.3389/fpls.2022.876357)
4. Masuda, K. et al. (2023). Application of deep learning diagnosis for multiple traits sorting in peach fruit. *Postharvest Biology and Technology*, 112348. [요약](https://bohrium.dp.tech/paper/arxiv/39372855) · [AMiner](https://aminer.org/pub/647f1924d68f896efa1c210d/application-of-deep-learning-diagnosis-for-multiple-traits-sorting-in-peach-fruit)
5. Masih, R. et al. (2026). 복숭아 잿빛무늬병·잎오갈병 현장 진단 CNN (DurAPP). *AgriEngineering*, 8, 368. [PDF](https://mdpi-res.com/d_attachment/agriengineering/agriengineering-08-00368/article_deploy/agriengineering-08-00368.pdf)
6. Identification of disease using deep learning and evaluation of bacteriosis in peach leaf (2021). *Ecological Informatics*. [X-MOL](https://www.x-mol.com/paper/1358610269332086784) · [Peeref](https://www.peeref.com/works/21721244)
7. 복숭아 과실 MobileNetV2 전이학습 분류 (2020). *International Conference on Decision Aid Sciences and Application (DASA)*. [PDF](https://www.di.ubi.pt/~hugomcp/doc/DASA_2020.pdf)
8. Leveraging YOLO deep learning models to enhance plant disease identification (2025). *Scientific Reports*. [PDF](https://rcastoragev2.blob.core.windows.net/6f713250f40f5cfe641129da9ded15a6/41598_2025_Article_92143.PMC11889226.pdf)
9. 개선 YOLOv8 기반 복숭아 잎오갈병 탐지 (2024). *Electronics*, 13(24), 4973. [MDPI](https://www.mdpi.com/2079-9292/13/24/4973)

### 근연 과수·방법 참고
10. 자두 현장 병해 탐지, 모바일 정확도 92% 이상 (2020). *Sensors*, 20, 5569. [PDF](https://mdpi-res.com/d_attachment/sensors/sensors-20-05569/article_deploy/sensors-20-05569.pdf)
11. Leveraging deep learning for plant disease and pest detection: a comprehensive review (2025). [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11885274/)

### 해충
12. Deep Learning Embedded into Smart Traps for Fruit Insect Pests Detection (복숭아순나방 포함). [링크](https://www.bohrium.com/paper-details/deep-learning-embedded-into-smart-traps-for-fruit-insect-pests-detection/864986920172650558-2616)
13. Ding, W., Taylor, G. (2016). Automatic Moth Detection from Trap Images for Pest Management. [arXiv](https://arxiv.org/pdf/1602.07383)
14. Research on a machine vision-based electro-killing pheromone-baited intelligent agricultural pest monitoring method (2025). [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11922902/)
15. Zhao et al. (2016). 복숭아순나방 트랩 영상 자동 계수 (영상처리). 검색 결과에서 원문 링크를 찾지 못함

### 과실 탐지 (1단계)
16. EMA-YOLO: 미숙 황도 탐지 (2024). *Sensors*, 24(12), 3783. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11207478)
17. PdYOLO: A Lightweight Algorithm for Detecting Peach Fruits Against a Peach Tree Background (2024). [링크](https://library.kab.ac.ug/Record/doaj-art-1227e84c19284e379fc1e929c710145d/Description)
18. YOLO-Peach: 어린 과실 계수용 경량 YOLOv8s (2024). *Agronomy*, 14(8), 1628. [MDPI](https://mdpi.com/2073-4395/14/8/1628)

### 국내
19. 김민기 외 (2017). 유기재배 복숭아 과원의 관리현황 및 병해충 발생 실태. *한국유기농업학회지*, 25(3), 603–617. [KoreaScience](https://koreascience.kr/article/JAKO201725650934533.page)
20. Shin, H.-D. 외 (2018). *Podosphaera pannosa* Causes Powdery Mildew and Rusty Spot on Peach Fruits from Korea. *한국균학회지*, 46(2), 193–199. [KoreaScience](https://koreascience.kr/article/JAKO201817968657849.page)
21. 합성곱 신경망을 이용한 '미황' 복숭아 과실의 성숙도 분류 (2022). *생물환경조절학회지*, 31. [KoreaScience](https://koreascience.kr/article/JAKO202232055723394.page?lang=ko) · [ScholarWorks](https://scholarworks.gnu.ac.kr/item/dc82a600-002d-43c4-9827-36e7c1ffbd44)
22. 딥러닝 기술을 활용한 복숭아 '미황'의 성숙도 자동 분류 (2024). *원예과학기술지*, 42(1). [PDF](https://www.hst-j.org/articles/pdf/1Qry/kshs-2024-042-01-7.pdf) · [KCI](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003052284)
23. 김진국 외 (2023). 복숭아 '미황' 과실 로봇 수확을 위한 숙도 자동분류 딥러닝 기술. *한국원예학회 학술발표요지*. [DBpia](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11414891)
24. 수확 후 보르도액 살포에 의한 복숭아 세균구멍병 방제 (2001). *식물병연구*. [KoreaScience](https://koreascience.kr/article/JAKO200111920832309.page)

### 데이터셋
- [Mendeley PDDICP — 복숭아 병해 현장 사진](https://data.mendeley.com/datasets/3pmj85snvw/1)
- [Roboflow Peach Disease (zeerox)](https://universe.roboflow.com/zeerox/peach-disease) · [Roboflow peach detection](https://universe.roboflow.com/islamia-college-peshawar-x56lw/peach-detection-rfv6n) · [Roboflow PEST DETECTION](https://universe.roboflow.com/yooseop-jang-a4fo7/pest-detection-jzeof)
- [Mendeley AgriVision-Apricot](https://data.mendeley.com/datasets/5wk5wpzczw/1) · [Mendeley Plum Leaf and Fruit (인도대추로 추정)](https://data.mendeley.com/datasets/w7sdx55m7z/1)
- AI Hub: [노지 작물 질병(147)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=147) · [노지 작물 해충(148)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=148) · [과수화상병(146)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=146) · [농산물 품질 QC(149)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=149)

> 저자를 "외"나 설명형 제목으로 적은 항목(5~11, 16~18, 21~22)은 검색 요약만 보고 쓴 것이다. 발표 자료에 인용하기 전에 원문에서 정확한 저자와 제목을 확인할 것.

## 8. 다음 할 일

- [x] Mendeley PDDICP 내려받아 클래스·장수·과실 사진 비율 확인 (5-1)
- [x] Roboflow 데이터 3종의 장수·라이선스 확인
- [x] Roboflow Islamia 데이터 다운로드 → PDDICP와 중복 검사 (2026-11-14, 50장 겹침. 실험 기록 실험 2 참고)
- [ ] Masuda 2023, L2MXception 2021, AgriEngineering 2026 원문에서 수치 확인
- [ ] 관행 재배 기준 국내 복숭아 병해 발생 통계(농사로·도 농업기술원 예찰 자료) 확인
- [ ] 팀 과원(미홍)에서 병해·피해과를 실제로 몇 개나 찍을 수 있는지 확인

🏆 Competition Projects Repository
머신러닝·딥러닝 기반 AI Competition 실험 저장소

이 저장소는 다양한 데이터 분석·머신러닝·딥러닝 기반 Competition에서 수행한
전처리 → 모델링 → 최적화 → 제출 파이프라인 전체 코드를 모아놓은 실험 저장소입니다.

각 프로젝트는 독립적으로 실행될 수 있도록 구성되어 있으며,
모듈화된 코드, 고급 Feature Engineering, 모델 튜닝, 앙상블 등을 포함합니다.

```text
📂 Repository Structure
Competition/
├── Future_Sales_Demographics/
├── SMILES-Based_Drug_Activity_Prediction/
├── Smart_Port_AGV_Route_Optimization/
├── Thyroid_Cancer_Classification/
├── Toss_CTR_Prediction/
└── README.md
```
---

🧾 1. Future Sales & Demographics Prediction

미래 판매량 & 인구통계 기반 예측 모델링

Time-Series Forecasting + Feature Fusion + ML 모델링 기반 문제

🔧 주요 기능

시계열 기반 판매량 예측

인구통계 정보 병합 Feature Engineering

Lag/Window 통계 피처 생성

CatBoost / LightGBM / XGBoost 기반 예측

모델 앙상블 및 OOF 기반 검증

제출 파일 자동 생성

🧠 기술 스택

Python, Pandas, Numpy, LightGBM, CatBoost, XGBoost, Scikit-Learn

---

🧬 2. SMILES-Based Drug Activity Prediction

화학 구조(SMILES) 기반 MAP3K5 IC50(pIC50) 활성 예측

RDKit + Chemprop(D-MPNN) + CatBoost 회귀 모델을 활용한 약물 활성 예측 프로젝트

🔧 주요 기능

CAS / ChEMBL / PubChem → 3-source 통합

Canonical SMILES 정규화 (RDKit)

RDKit Descriptor & Fingerprint 생성

Chemprop D-MPNN Embedding

High-activity Oversampling

CatBoost 회귀 모델

GroupKFold 기반 검증 및 제출 생성
```text
📁 포함 파일
smiles_data.py
smiles_data_go.py
smiles_model_final.py
smiles_model_go.py
```
🧠 기술 스택

Python, RDKit, Chemprop, PyTorch Lightning, CatBoost, Scikit-Learn

---

🚢 3. Smart Port AGV Route Optimization

스마트 항만 AGV(Automated Guided Vehicle) 최적 경로 탐색

OR-Tools 기반 초기 솔루션 + Local Search + ALNS 기반 최적화 엔진

🔧 주요 기능

OR-Tools Initial Solution 생성

2-Opt / Shaw Removal / Regret Insertion / Worst Removal

ALNS (Adaptive Large Neighborhood Search)

Elite Set 기반 재구성(Reconstruct)

Operator weight 업데이트
```text
📁 포함 파일
agv_00.py
agv_01_local_search_solver_20pt.py
agv_01_local_search_solver_ALNS_WorstRemoval.py
agv_01_local_search_solver_RegretInsertion.py
agv_01_local_search_solver_fast.py
agv_01_local_search_solver_shaw.py
agv_02_reconstruct_from_elites.py
```
🧠 기술 스택

Python, OR-Tools, Numpy, Pandas

---

🩺 4. Thyroid Cancer Classification [ 최종 1위 수상 ]

갑상선암 binary classification 문제

CatBoostEncoder + LightGBM / XGBoost / CatBoost 앙상블 기반 구조

🔧 주요 기능

CatBoostEncoder 기반 범주형 인코딩

LightGBM Optuna 튜닝

LightGBM + XGBoost + CatBoost Soft Voting Ensemble

F1 Score 최대 threshold 자동 탐색

Fold별 모델 저장

Feature Importance 시각화
```text
📁 포함 파일
thyroid_1000.py
thyroid_2000.py
thyroid_catencoder.py
thyroid_ensemble.py
thyroid_final.py
```
🧠 기술 스택

Python, LightGBM, XGBoost, CatBoost, Optuna, Scikit-Learn

---

📈 5. Toss CTR Prediction

대규모 광고 데이터셋 기반 CTR 예측 모델

PyArrow Parquet Streaming, 고급 Feature Engineering,
GPU XGBoost + FT-Transformer 기반 딥러닝 실험

🔧 주요 기능

PyArrow Parquet Batch Streaming 로딩

Frequency Encoding

Group Aggregation / Interaction Features

OOF Smoothed Target Encoding

GPU-XGBoost 기반 AUC-PR 최적화

FT-Transformer 실험

10-Fold OOF → Test Ensemble

자동 제출 생성
```text
📁 포함 파일
CTR_Feature Engineering.py
CTR_stop100.py
CTR_FT-Transformer.py
CTR_final.py
```
🧠 기술 스택

Python, PyArrow, Pandas, Numpy, XGBoost GPU, PyTorch, tqdm, gc

---
```text
🛠 전체 기술 스택 요약
Languages: Python  
Machine Learning: LightGBM, CatBoost, XGBoost  
Deep Learning: PyTorch, Chemprop (D-MPNN)  
Optimization: OR-Tools, ALNS  
Data Engineering: Pandas, Numpy, PyArrow  
Cheminformatics: RDKit  
Hyperparameter Search: Optuna  
Visualization & Utility: Matplotlib, Seaborn, tqdm, gc, joblib
```

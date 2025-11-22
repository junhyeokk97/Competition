# Thyroid Cancer Classification (Binary Prediction)

## 프로젝트 개요

갑상선암 진단 분류 해커톤

갑상선암 이진 분류 대회를 위한 AI 모델링 파이프라인을 구축하는 프로젝트입니다.

## 주요 기능

- OrdinalEncoder과 LightGBM
- Optuna 하이퍼파라미터 최적화
- OOF 기반 K-Fold 학습 및 성능 평가

## 파일 구조

```
Thyroid_Cancer_Classification/
├── thyroid_catencoder.py      # CatBoostEncoder + LightGBM 모델      (전처리 강화)
├── thyroid_ensemble.py        # LGBM / XGB / CatBoost 앙상블 모델     (3-Model Voting)
├── thyroid_1000.py            # LGBM Optuna (1000 n_est) 기본 버전
├── thyroid_2000.py            # LGBM Optuna (2000 n_est) 확장 버전
├── thyroid_final.py           # LGBM 고난이도 탐색 버전 (200 Trials)
└── README.md  
```

## 기술 스택

- Python
- LightGBM, CatBoost/CatBoostEncoder, XGBoost
- Optuna, Scikit-Learn
- pandas, numpy
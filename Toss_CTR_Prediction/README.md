# Toss CTR Prediction

## 프로젝트 개요

토스 NEXT ML CHALLENGE : 광고 클릭 예측(CTR) 모델 개발

대규모 온라인 광고 클릭률(CTR) 예측 문제를 해결하기 위해 다양한 모델링 프로젝트입니다.

## 주요 기능

- 대규모 데이터 최적화 로딩 (PyArrow, Batch Streaming)
- CTR 특화 Feature Engineering

## 파일 구조

```
Toss_CTR_Prediction/
├── CTR_Feature Engineering.py       # 대규모 FE + XGBoost 파이프라인
├── CTR_stop100.py                   # Early stopping 및 하이퍼파라미터 개선 버전
├── CTR_FT-Transformer.py            # FT-Transformer 딥러닝 학습 코드
├── CTR_final.py                     # 최종 XGBoost 기반 제출 파이프라인
└── README.md  
```

## 기술 스택

- Python
- XGBoost
- pyarrow, parquet
- pandas, numpy
# SMILES-Based Drug Activity Prediction

## 프로젝트 개요

Jump AI(.py) 2025 : 제 3회 AI 신약개발 경진대회:MAP3K5 (ASK1) IC50 Prediction Competition

약물 분자의 SMILES 문자열을 기반으로 IC50(nM) 값을 정확하게 예측하는 모델링 프로젝트입니다.

## 주요 기능

- 데이터 정제 및 통합
- 분자 특징 (Feature Engineering)
- 데이터 증강 (Augmentation)

## 파일 구조

```
smiles-modeling/
├── smiles_data.py                # CAS / ChEMBL / PubChem 데이터 통합 및 전처리
├── smiles_data_go.py             # 동일 전처리 파이프라인 (경로/구조 변경 버전)
├── smiles_model_final.py         # Chemprop + CatBoost 결합 최종 모델
├── smiles_model_go.py            # 경로 변경 및 흐름 개선 버전
└── README.md  
```

## 기술 스택

- Pytorch
- RDKit
- pandas, numpy
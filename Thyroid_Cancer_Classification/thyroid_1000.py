import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# =======================
# 1. 데이터 불러오기 및 전처리
# =======================
path = './data/cancer/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

label_cols = ['Gender', 'Country', 'Race', 'Family_Background', 
              'Radiation_History', 'Iodine_Deficiency', 
              'Smoke', 'Weight_Risk', 'Diabetes']

# train+test 합쳐서 인코딩 (학습 데이터에 없는 범주도 대비)
full_data = pd.concat([train.drop(columns='Cancer'), test])
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
full_data[label_cols] = encoder.fit_transform(full_data[label_cols])

x = full_data.iloc[:len(train)]
x_test = full_data.iloc[len(train):]
y = train['Cancer']

# =======================
# 3. Optuna 목적 함수 정의 (튜닝 대상: max_depth, num_leaves, learning_rate 등)
# =======================
def objective(trial):
    params = {
        'n_estimators': 1000,
        'max_depth': trial.suggest_int('max_depth', 25, 2000),  # 탐색 범위 확대
        'num_leaves': trial.suggest_int('num_leaves', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.09, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 55),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }
    f1_scores = []
    for train_idx, val_idx in skf.split(x, y):
        X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='binary_error',  # binary_logloss -> binary_error (분류 오류율)
                  callbacks=[early_stopping(stopping_rounds=100, verbose=False)])  # early stopping rounds 증가

        val_prob = model.predict_proba(X_val)[:, 1]

        # 최적 임계값 찾기 (0.1~0.9 범위)
        best_f1 = 0
        for thr in np.arange(0.1, 0.9, 0.01):
            val_pred_thr = (val_prob > thr).astype(int)
            f1 = f1_score(y_val, val_pred_thr)
            if f1 > best_f1:
                best_f1 = f1
        f1_scores.append(best_f1)

    return np.mean(f1_scores)

# =======================
# 4. Optuna 스터디 생성 및 튜닝 (샘플러 고정, 100 trials로 증가)
# =======================
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best trial:")
trial = study.best_trial
print(f"  F1 Score: {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
    
# =======================
# 5. 최적 파라미터 설정 (튜닝 결과 + 기본 설정 보완)
# =======================
best_params = trial.params
best_params.update({
    'n_estimators': 1000,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
})

# =======================
# 6. 최종 모델 학습 (5-fold)
# =======================
final_models = []
thresholds = []
scores = []

for train_idx, val_idx in skf.split(x, y):
    X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='binary_error',
              callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
              )

    final_models.append(model)

    val_prob = model.predict_proba(X_val)[:, 1]
    best_f1 = 0
    best_thr = 0.5
    for thr in np.arange(0.1, 0.9, 0.01):
        val_pred_thr = (val_prob > thr).astype(int)
        f1 = f1_score(y_val, val_pred_thr)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    thresholds.append(best_thr)
    scores.append(best_f1)
    
# =======================
# 7. 성능 기준 가장 좋은 fold 선택 (가중 평균 등으로 발전 가능)
# =======================
best_index = np.argmax(scores)
final_model = final_models[best_index]
final_thr = thresholds[best_index]

print(f"Best fold index: {best_index}")
print(f"Best fold F1 score: {scores[best_index]:.4f}")
print(f"Best threshold: {final_thr:.2f}")

# =======================
# 8. 테스트 데이터 예측 (5개 fold 모델 예측 평균)
#    - 가중 평균으로 확장 가능
# =======================
test_prob = np.mean([model.predict_proba(x_test)[:, 1] for model in final_models], axis=0)
test_pred = (test_prob > final_thr).astype(int)

submission['Cancer'] = test_pred
submission.to_csv(path + 'sub_1.csv')
print("저장.")

# =======================
# 9. 피처 중요도 시각화
# =======================
feat_imp = pd.Series(final_model.feature_importances_, index=x.columns)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importance (LightGBM)")
plt.tight_layout()
plt.show()

# =======================
# 10. 모델 저장 (각 fold 모델 + 임계값 + 최적 파라미터)
# =======================
model_dir = './data/cancer/'
os.makedirs(model_dir, exist_ok=True)

for i, model in enumerate(final_models):
    model_path = os.path.join(model_dir, f'lgbm_fold{i}.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")
    
thresholds_path = os.path.join(model_dir, 'thresholds.npy')
joblib.dump(thresholds, thresholds_path)
print(f"Thresholds saved: {thresholds_path}")

params_path = os.path.join(model_dir, 'best_params.pkl')
joblib.dump(best_params, params_path)
print(f"Best parameters saved: {params_path}")

best_model_path = os.path.join(model_dir, 'lgbm_best_fold.pkl')
joblib.dump(final_models[best_index], best_model_path)
print(f"Best model saved: {best_model_path}")
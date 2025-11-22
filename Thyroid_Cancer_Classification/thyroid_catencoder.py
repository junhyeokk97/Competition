import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from category_encoders import CatBoostEncoder

# =======================
# 1. 데이터 불러오기
# =======================
path = './data/cancer/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

label_cols = [
    'Gender', 'Country', 'Race', 'Family_Background',
    'Radiation_History', 'Iodine_Deficiency',
    'Smoke', 'Weight_Risk', 'Diabetes'
]

# =======================
# 2. CatBoostEncoder 기반 전처리
# =======================
train_x = train.drop(columns='Cancer')
train_y = train['Cancer']
test_x = test.copy()

# CatBoostEncoder는 train에만 fit → leakage 방지
cbe = CatBoostEncoder(cols=label_cols, random_state=42)
train_x[label_cols] = cbe.fit_transform(train_x[label_cols], train_y)
test_x[label_cols] = cbe.transform(test_x[label_cols])

x = train_x
x_test = test_x
y = train_y.values

# =======================
# 3. Stratified K-Fold
# =======================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =======================
# 4. Optuna 목적 함수 정의
# =======================
def objective(trial):
    params = {
        'n_estimators': 2000,
        'max_depth': trial.suggest_int('max_depth', 25, 330),
        'num_leaves': trial.suggest_int('num_leaves', 150, 300),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }

    f1_scores = []
    for tr_idx, val_idx in skf.split(x, y):
        X_tr, X_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_error',
            callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
        )

        val_prob = model.predict_proba(X_val)[:, 1]

        # threshold tuning
        best_f1 = 0
        for thr in np.arange(0.1, 0.9, 0.01):
            pred_thr = (val_prob > thr).astype(int)
            f1 = f1_score(y_val, pred_thr)
            if f1 > best_f1:
                best_f1 = f1

        f1_scores.append(best_f1)

    return np.mean(f1_scores)

# =======================
# 5. Optuna 튜닝 실행
# =======================
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Best trial:")
trial = study.best_trial
print(f"  F1 Score: {trial.value:.4f}")
print("  Params:")
for k, v in trial.params.items():
    print(f"    {k}: {v}")

# =======================
# 6. 최적 파라미터 적용
# =======================
best_params = trial.params
best_params.update({
    'n_estimators': 2000,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
})

# =======================
# 7. 최종 5-Fold 학습
# =======================
final_models = []
thresholds = []
scores = []

for tr_idx, val_idx in skf.split(x, y):
    X_tr, X_val = x.iloc[tr_idx], x.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = LGBMClassifier(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_error',
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
    )

    final_models.append(model)

    val_prob = model.predict_proba(X_val)[:, 1]

    # threshold search
    best_f1 = 0
    best_thr = 0.5
    for thr in np.arange(0.1, 0.9, 0.01):
        pred_thr = (val_prob > thr).astype(int)
        f1 = f1_score(y_val, pred_thr)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    thresholds.append(best_thr)
    scores.append(best_f1)

# =======================
# 8. best fold
# =======================
best_index = np.argmax(scores)
final_model = final_models[best_index]
final_thr = thresholds[best_index]

print(f"Best fold index: {best_index}")
print(f"Best fold F1: {scores[best_index]:.4f}")
print(f"Best threshold: {final_thr:.2f}")

# =======================
# 9. 테스트 예측
# =======================
test_prob = np.mean(
    [m.predict_proba(x_test)[:, 1] for m in final_models],
    axis=0
)
test_pred = (test_prob > final_thr).astype(int)

submission['Cancer'] = test_pred
submission.to_csv(path + 'sub_catboostencoder_lgbm.csv')
print("저장 완료.")

# =======================
# 10. Feature Importance
# =======================
feat_imp = pd.Series(final_model.feature_importances_, index=x.columns)
plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importance (LightGBM)")
plt.tight_layout()
plt.show()

# =======================
# 11. 모델 저장
# =======================
save_dir = './data/cancer/models_catboostencoder/'
os.makedirs(save_dir, exist_ok=True)

# fold 모델 저장
for i, model in enumerate(final_models):
    joblib.dump(model, os.path.join(save_dir, f'lgbm_fold{i}.pkl'))

# threshold & params 저장
joblib.dump(thresholds, os.path.join(save_dir, 'thresholds.pkl'))
joblib.dump(best_params, os.path.join(save_dir, 'best_params.pkl'))
joblib.dump(final_models[best_index], os.path.join(save_dir, 'lgbm_best_fold.pkl'))

print("모델 저장 완료.")

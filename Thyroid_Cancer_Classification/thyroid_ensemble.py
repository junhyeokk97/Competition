import os
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score

from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# =======================
# 1. 데이터 불러오기 및 전처리
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

# train + test 합쳐서 인코딩 (train에 없는 범주 대비)
full_data = pd.concat([train.drop(columns='Cancer'), test])
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
full_data[label_cols] = encoder.fit_transform(full_data[label_cols])

x = full_data.iloc[:len(train)]
x_test = full_data.iloc[len(train):]
y = train['Cancer'].values

# 공통 K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# =======================
# 2. 공통 유틸 함수 정의
# =======================
def find_best_threshold(y_true, prob, min_thr=0.1, max_thr=0.9, step=0.01):
    """F1 score 기준으로 best threshold 탐색"""
    best_f1 = 0.0
    best_thr = 0.5
    for thr in np.arange(min_thr, max_thr, step):
        pred = (prob > thr).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


# =======================
# 3. Optuna 목적 함수들 정의
#    - LGBM / XGB / CatBoost 각각 튜닝
# =======================
def objective_lgbm(trial):
    params = {
        'n_estimators': 1000,
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
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
            eval_metric='binary_logloss',
            callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
        )

        val_prob = model.predict_proba(X_val)[:, 1]
        _, best_f1 = find_best_threshold(y_val, val_prob)
        f1_scores.append(best_f1)

    return float(np.mean(f1_scores))


def objective_xgb(trial):
    params = {
        'n_estimators': 1000,
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'scale_pos_weight': 1.0,  # class imbalance는 일단 LGBM처럼 맞춰도 됨
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }

    f1_scores = []
    for tr_idx, val_idx in skf.split(x, y):
        X_tr, X_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=100
        )

        val_prob = model.predict_proba(X_val)[:, 1]
        _, best_f1 = find_best_threshold(y_val, val_prob)
        f1_scores.append(best_f1)

    return float(np.mean(f1_scores))


def objective_cat(trial):
    params = {
        'iterations': 1000,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_state': 42,
        'eval_metric': 'Logloss',
        'loss_function': 'Logloss',
        'verbose': False
    }

    f1_scores = []
    for tr_idx, val_idx in skf.split(x, y):
        X_tr, X_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            use_best_model=True,
            early_stopping_rounds=100,
        )

        val_prob = model.predict_proba(X_val)[:, 1]
        _, best_f1 = find_best_threshold(y_val, val_prob)
        f1_scores.append(best_f1)

    return float(np.mean(f1_scores))


# =======================
# 4. 모델별 Optuna 튜닝
# =======================
sampler = optuna.samplers.TPESampler(seed=42)

print("=== [1/3] LightGBM 튜닝 중 ===")
study_lgbm = optuna.create_study(direction='maximize', sampler=sampler)
study_lgbm.optimize(objective_lgbm, n_trials=30, show_progress_bar=True)
best_params_lgbm = study_lgbm.best_trial.params
best_params_lgbm.update({
    'n_estimators': 1000,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
})
print("Best LGBM params:", best_params_lgbm)

print("\n=== [2/3] XGBoost 튜닝 중 ===")
study_xgb = optuna.create_study(direction='maximize', sampler=sampler)
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)
best_params_xgb = study_xgb.best_trial.params
best_params_xgb.update({
    'n_estimators': 1000,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
})
print("Best XGB params:", best_params_xgb)

print("\n=== [3/3] CatBoost 튜닝 중 ===")
study_cat = optuna.create_study(direction='maximize', sampler=sampler)
study_cat.optimize(objective_cat, n_trials=30, show_progress_bar=True)
best_params_cat = study_cat.best_trial.params
best_params_cat.update({
    'iterations': 1000,
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'random_state': 42,
    'verbose': False
})
print("Best CatBoost params:", best_params_cat)


# =======================
# 5. 최종 5-Fold 학습 (3-모델 × 5-Fold)
# =======================
lgbm_models = []
xgb_models = []
cat_models = []
fold_thresholds = []
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(x, y), 1):
    print(f"\n===== Fold {fold} =====")

    X_tr, X_val = x.iloc[tr_idx], x.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # --- LGBM ---
    lgbm = LGBMClassifier(**best_params_lgbm)
    lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
    )
    lgbm_models.append(lgbm)

    # --- XGBoost ---
    xgb = XGBClassifier(**best_params_xgb)
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=100
    )
    xgb_models.append(xgb)

    # --- CatBoost ---
    cat = CatBoostClassifier(**best_params_cat)
    cat.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        use_best_model=True,
        early_stopping_rounds=100,
    )
    cat_models.append(cat)

    # --- 앙상블 예측 (soft voting) ---
    val_prob_lgbm = lgbm.predict_proba(X_val)[:, 1]
    val_prob_xgb = xgb.predict_proba(X_val)[:, 1]
    val_prob_cat = cat.predict_proba(X_val)[:, 1]

    val_prob_ensemble = (val_prob_lgbm + val_prob_xgb + val_prob_cat) / 3.0

    best_thr, best_f1 = find_best_threshold(y_val, val_prob_ensemble)
    fold_thresholds.append(best_thr)
    fold_scores.append(best_f1)

    print(f"Fold {fold} | Best F1: {best_f1:.4f} | Best thr: {best_thr:.3f}")

# 최종 threshold: fold별 평균 사용 (원하면 최대 F1 fold의 thr 사용도 가능)
final_threshold = float(np.mean(fold_thresholds))
best_fold_idx = int(np.argmax(fold_scores))
print("\n===== 최종 결과 =====")
print(f"Fold별 F1: {fold_scores}")
print(f"평균 F1: {np.mean(fold_scores):.4f}")
print(f"최고 F1 fold index: {best_fold_idx}")
print(f"최종 사용 threshold(평균): {final_threshold:.3f}")


# =======================
# 6. 테스트 데이터 예측 (3-모델 × 5-Fold 앙상블)
# =======================
test_prob_lgbm = np.mean(
    [m.predict_proba(x_test)[:, 1] for m in lgbm_models],
    axis=0
)
test_prob_xgb = np.mean(
    [m.predict_proba(x_test)[:, 1] for m in xgb_models],
    axis=0
)
test_prob_cat = np.mean(
    [m.predict_proba(x_test)[:, 1] for m in cat_models],
    axis=0
)

test_prob_ensemble = (test_prob_lgbm + test_prob_xgb + test_prob_cat) / 3.0
test_pred = (test_prob_ensemble > final_threshold).astype(int)

submission['Cancer'] = test_pred
sub_path = os.path.join(path, 'sub_ensemble_lgbm_xgb_cat.csv')
submission.to_csv(sub_path)
print(f"\n제출 파일 저장 완료: {sub_path}")


# =======================
# 7. Feature Importance (LightGBM 기준)
# =======================
feat_imp = pd.Series(lgbm_models[best_fold_idx].feature_importances_, index=x.columns)
feat_imp = feat_imp.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importance (LightGBM)")
plt.tight_layout()
plt.show()


# =======================
# 8. 모델 및 메타 정보 저장
# =======================
model_dir = './data/cancer/models_ensemble/'
os.makedirs(model_dir, exist_ok=True)

# 개별 fold 모델 저장
for i, (m_lgbm, m_xgb, m_cat) in enumerate(zip(lgbm_models, xgb_models, cat_models)):
    joblib.dump(m_lgbm, os.path.join(model_dir, f'lgbm_fold{i}.pkl'))
    joblib.dump(m_xgb, os.path.join(model_dir, f'xgb_fold{i}.pkl'))
    joblib.dump(m_cat, os.path.join(model_dir, f'cat_fold{i}.pkl'))

joblib.dump(fold_thresholds, os.path.join(model_dir, 'fold_thresholds.pkl'))
joblib.dump(final_threshold, os.path.join(model_dir, 'final_threshold.pkl'))

joblib.dump(best_params_lgbm, os.path.join(model_dir, 'best_params_lgbm.pkl'))
joblib.dump(best_params_xgb, os.path.join(model_dir, 'best_params_xgb.pkl'))
joblib.dump(best_params_cat, os.path.join(model_dir, 'best_params_cat.pkl'))

print(f"\n모델 및 파라미터 저장 완료: {model_dir}")

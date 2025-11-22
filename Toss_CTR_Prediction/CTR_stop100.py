import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import torch
from datetime import datetime
import pyarrow.parquet as pq
import gc

# 모델 라이브러리
import xgboost as xgb

# ===================================================================
# 1. 설정 (CONFIGURATIONS)
# ===================================================================
class CFG:
    # --- 기본 설정 ---
    SEED = 7
    N_FOLDS = 10
    TARGET_0_RATIO = 1.5

    # --- 경로 설정 ---
    DATA_PATH = './data/toss/data/'
    SAVE_PATH = './data/toss/save/'
    TRAIN_PATH = os.path.join(DATA_PATH, "train.parquet")
    TEST_PATH = os.path.join(DATA_PATH, "test.parquet")
    SUBMISSION_PATH = os.path.join(DATA_PATH, 'sample_submission.csv')

    # --- 데이터 로딩 및 샘플링 ---
    BATCH_SIZE = 200000

    # --- 피처 엔지니어링 ---
    TARGET_COL = "clicked"
    MAIN_CAT_FEATS = ['inventory_id', 'age_group', 'ad_id', 'carrier', 'platform', 'gender', 'day_of_week']
    MAIN_NUM_FEATS = ['history_a_1', 'l_feat_16', 'l_feat_1', 'history_b_2', 'l_feat_2', 'history_a_2', 'history_b_1']
    INTERACTION_PAIRS = [['inventory_id', 'age_group'], ['ad_id', 'platform'], ['ad_id', 'age_group']]
    EPSILON = 1e-6

    # --- 모델 훈련 ---
    SMOOTHING_FACTOR = 20
    XGB_PARAMS = {
        'objective': 'binary:logistic', 'eval_metric': 'aucpr',
        'tree_method': 'hist', 'device': 'cuda', 'random_state': SEED,
        'learning_rate': 0.01, 'max_depth': 4, 'subsample': 0.77,
        'colsample_bytree': 0.7, 'gamma': 0.03, 'lambda': 0.8, 'alpha': 0.1,
        'enable_categorical': True
    }
    NUM_BOOST_ROUND = 10000
    EARLY_STOPPING_ROUNDS = 100

# ===================================================================
# 2. 유틸리티 (Utilities)
# ===================================================================
def log_message(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
                else: df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: log_message(f'Memory usage reduced to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# ===================================================================
# 3. 메인 로직 (Main Logic)
# ===================================================================
seed_everything(CFG.SEED)
os.makedirs(CFG.SAVE_PATH, exist_ok=True)

log_message("테스트 데이터 로딩...")
test = pd.read_parquet(CFG.TEST_PATH, engine="pyarrow").drop(columns=['ID'])

log_message("훈련 데이터 샘플링 시작...")
pf = pq.ParquetFile(CFG.TRAIN_PATH)
n_rows = pf.metadata.num_rows
n_clicked_1 = 0
for chunk in tqdm(pf.iter_batches(batch_size=CFG.BATCH_SIZE, columns=['clicked']), total=-(-n_rows//CFG.BATCH_SIZE), desc="Target 카운팅"):
    n_clicked_1 += chunk.to_pandas()['clicked'].sum()

n_clicked_0 = n_rows - n_clicked_1
target_0_samples = n_clicked_1 * CFG.TARGET_0_RATIO
sample_ratio_0 = target_0_samples / n_clicked_0 if n_clicked_0 > 0 else 0

sampled_chunks = []
for chunk in tqdm(pf.iter_batches(batch_size=CFG.BATCH_SIZE), total=-(-n_rows//CFG.BATCH_SIZE), desc="데이터 로딩 및 샘플링"):
    df_chunk = chunk.to_pandas()
    df_1 = df_chunk[df_chunk['clicked'] == 1]
    df_0 = df_chunk[df_chunk['clicked'] == 0]
    df_0_sampled = df_0.sample(frac=sample_ratio_0, random_state=CFG.SEED)
    sampled_chunks.append(pd.concat([df_1, df_0_sampled]))

train_df = pd.concat(sampled_chunks, ignore_index=True).sample(frac=1, random_state=CFG.SEED).reset_index(drop=True)
del pf, sampled_chunks, df_chunk, df_1, df_0, df_0_sampled
gc.collect()

log_message("피처 엔지니어링 시작...")
FEATURE_EXCLUDE = {CFG.TARGET_COL, "seq", "ID"}
train_len = len(train_df)

y_target = train_df[CFG.TARGET_COL].copy()
combined_df = pd.concat([train_df.drop(columns=[CFG.TARGET_COL]), test], ignore_index=True)

combined_df['hour'] = pd.to_numeric(combined_df['hour'], errors='coerce').astype(int)
combined_df['time_of_day'] = pd.cut(combined_df['hour'], bins=[-1, 6, 12, 18, 23], labels=[0, 1, 2, 3], ordered=False).astype(int)

existing_cat_feats = [col for col in CFG.MAIN_CAT_FEATS if col in combined_df.columns]
existing_num_feats = [col for col in CFG.MAIN_NUM_FEATS if col in combined_df.columns]

# ### [개선 3] 빈도 인코딩 (Frequency Encoding) ###
log_message("빈도 인코딩 피처 생성...")
for col in tqdm(existing_cat_feats, desc="빈도 인코딩"):
    freq_map = combined_df[col].value_counts(normalize=True)
    combined_df[f'{col}_freq'] = combined_df[col].map(freq_map)
    
log_message("확장된 그룹 통계 피처 생성...")
aggs = ['mean', 'std', 'min', 'max', 'median', 'nunique']
for cat_feat in tqdm(existing_cat_feats, desc="범주형 피처 루프"):
    for num_feat in existing_num_feats:
        for agg in aggs:
            group_agg = combined_df.groupby(cat_feat)[num_feat].transform(agg)
            combined_df[f'{cat_feat}_{num_feat}_{agg}'] = group_agg
        combined_df[f'{cat_feat}_{num_feat}_mean_diff'] = combined_df[num_feat] - combined_df[f'{cat_feat}_{num_feat}_mean']

log_message("고차원 상호작용 피처 생성...")
for pair in tqdm(CFG.INTERACTION_PAIRS, desc="고차원 상호작용 루프"):
    if all(c in combined_df.columns for c in pair):
        group_col_name = '_'.join(pair)
        combined_df[group_col_name] = combined_df[pair[0]].astype(str) + '_' + combined_df[pair[1]].astype(str)
        for num_feat in existing_num_feats:
            group = combined_df.groupby(group_col_name)[num_feat]
            combined_df[f'{group_col_name}_{num_feat}_mean'] = group.transform('mean')
            combined_df[f'{group_col_name}_{num_feat}_std'] = group.transform('std')
        combined_df.drop(columns=[group_col_name], inplace=True)

log_message("기본 상호작용 및 비율 피처 생성...")
if 'history_a_1' in combined_df.columns and 'history_b_2' in combined_df.columns:
    combined_df['history_a1_div_b2'] = combined_df['history_a_1'] / (combined_df['history_b_2'] + CFG.EPSILON)
if 'l_feat_1' in combined_df.columns and 'l_feat_2' in combined_df.columns:
    combined_df['l_feat_1_div_2'] = combined_df['l_feat_1'] / (combined_df['l_feat_2'] + CFG.EPSILON)

train_df = combined_df[:train_len].copy()
test = combined_df[train_len:].copy()
train_df[CFG.TARGET_COL] = y_target
del combined_df, y_target
gc.collect()

log_message("데이터 타입 변환 (object -> category)...")
categorical_cols = [c for c in train_df.columns if train_df[c].dtype == 'object']
for col in tqdm(categorical_cols):
    train_df[col] = train_df[col].astype('category')
    test[col] = test[col].astype('category')

log_message("결측치 처리 강화...")
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove(CFG.TARGET_COL) # 타겟 컬럼은 제외
for col in tqdm(numeric_cols, desc="결측치 중앙값 대체"):
    median_val = train_df[col].median()
    train_df[col].fillna(median_val, inplace=True)
    test[col].fillna(median_val, inplace=True)

# ### [개선 5] OOF 스무딩 타겟 인코딩 (OOF Smoothing Target Encoding) ###
log_message("OOF 스무딩 타겟 인코딩 시작...")
skf_te = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
global_mean = train_df[CFG.TARGET_COL].mean()

for col in tqdm(existing_cat_feats, desc="OOF 타겟 인코딩"):
    te_col_name = f'{col}_te'
    train_df[te_col_name] = np.nan

    # 훈련 데이터에 대한 OOF 인코딩 값 계산
    for train_idx, val_idx in skf_te.split(train_df, train_df[CFG.TARGET_COL]):
        X_train, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        
        grouped = X_train.groupby(X_train[col].astype(str))[CFG.TARGET_COL]
        means = grouped.mean()
        counts = grouped.count()
        
        smoothed_means = (counts * means + CFG.SMOOTHING_FACTOR * global_mean) / (counts + CFG.SMOOTHING_FACTOR)
        
        train_df.loc[val_idx, te_col_name] = X_val[col].astype(str).map(smoothed_means)

    train_df[te_col_name].fillna(global_mean, inplace=True)

    # 테스트 데이터 인코딩을 위해 전체 훈련 데이터를 사용
    grouped_full = train_df.groupby(train_df[col].astype(str))[CFG.TARGET_COL]
    means_full = grouped_full.mean()
    counts_full = grouped_full.count()
    smoothed_means_full = (counts_full * means_full + CFG.SMOOTHING_FACTOR * global_mean) / (counts_full + CFG.SMOOTHING_FACTOR)
    
    test[te_col_name] = test[col].astype(str).map(smoothed_means_full).fillna(global_mean)

log_message("OOF 스무딩 타겟 인코딩 완료.")
    
# ### [개선 4] 검증 전략에 대한 고찰 ###
# 현재 데이터셋에는 user_id와 같이 사용자를 특정할 수 있는 명확한 그룹핑 컬럼이 없습니다.
# 이러한 경우, StratifiedKFold는 각 Fold의 타겟 변수(clicked) 비율을 전체 데이터셋과 유사하게 유지시켜주므로,
# 클래스 불균형 상황에서 안정적이고 신뢰도 높은 검증 성능을 측정할 수 있는 매우 효과적인 방법입니다.
# 따라서 현재의 검증 전략을 유지합니다.

log_message("피처 엔지니어링 완료.")

# 사용할 모든 피처를 정의합니다.
feature_cols = [c for c in train_df.columns if c not in FEATURE_EXCLUDE]


# ===================================================================
# 4. 모델 훈련 및 제출 (Model Training & Submission)
# ===================================================================
if __name__ == '__main__':
    y = train_df[CFG.TARGET_COL].copy()
    X = train_df[feature_cols].copy()
    del train_df
    gc.collect()

    log_message("XGBoost 모델 훈련 시작...")

    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    models = []
    oof_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        log_message(f"======== Fold {fold}/{CFG.N_FOLDS} 훈련 시작 ========")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # OOF 타겟 인코딩이 사전에 완료되었으므로, 여기서는 추가적인 인코딩이 필요 없습니다.
        
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        
        model_params = CFG.XGB_PARAMS.copy()
        model_params['scale_pos_weight'] = (len(y_train) - y_train.sum()) / (y_train.sum() + CFG.EPSILON)
        
        model = xgb.train(
            params=model_params, dtrain=dtrain, num_boost_round=CFG.NUM_BOOST_ROUND,
            evals=[(dval, 'val')], early_stopping_rounds=CFG.EARLY_STOPPING_ROUNDS, verbose_eval=500
        )
        
        oof_preds = model.predict(dval, iteration_range=(0, model.best_iteration))
        fold_score = average_precision_score(y_val, oof_preds)
        oof_scores.append(fold_score)
        log_message(f"Fold {fold} AUC PR Score: {fold_score:.5f}")
        
        models.append(model)
        del dtrain, dval, X_train, y_train, X_val, y_val
        gc.collect()

    mean_oof_score = np.mean(oof_scores)
    log_message(f"📈 Mean OOF AUC PR Score (로컬 점수): {mean_oof_score:.5f}")

    log_message("테스트 데이터 예측 및 제출 파일 생성 시작...")
    
    # 테스트 데이터에 대한 타겟 인코딩이 사전에 완료되었습니다.
    model_feature_names = models[0].feature_names
    test_processed = test[model_feature_names]

    dtest_xgb = xgb.DMatrix(test_processed, enable_categorical=True)
    xgb_preds_list = [model.predict(dtest_xgb, iteration_range=(0, model.best_iteration)) for model in models]
    final_predictions = np.mean(xgb_preds_list, axis=0)
    
    submission = pd.read_csv(CFG.SUBMISSION_PATH)
    submission[CFG.TARGET_COL] = final_predictions
    
    current_time_str = datetime.now().strftime("%m%d_%H%M")
    submission_filename = f"submission_{current_time_str}_xgb_v2_oof_{mean_oof_score:.5f}.csv"
    submission_filepath = os.path.join(CFG.SAVE_PATH, submission_filename)
    
    submission.to_csv(submission_filepath, index=False)
    log_message(f"✅ 제출 파일 생성이 성공적으로 완료되었습니다: {submission_filepath}")


#📈 Mean OOF AUC PR Scoe (로컬 점수): 0.67292
#테스트 데이터 예측 및 제출 파일 생성 시작...
#✅ 제출 파일 생성이 성공적으로 완료되었습니다: ./Dacon/toss/_save/submission_1011_1643_xgb_v2_oof_0.67292.csv
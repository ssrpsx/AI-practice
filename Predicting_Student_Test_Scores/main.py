# ============================================================================
# Kaggle Playground Series S6E1: Student Test Score Prediction
# XGBoost with Ridge Meta-Feature (46 Features - OPTIMIZED)
# ============================================================================

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("="*80)
print("LOADING DATA")
print("="*80)

train_file = "/kaggle/input/playground-series-s6e1/train.csv"
test_file = "/kaggle/input/playground-series-s6e1/test.csv"
original_file = "/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
original_df = pd.read_csv(original_file)
submission_df = pd.read_csv("/kaggle/input/playground-series-s6e1/sample_submission.csv")

print(f"Train shape:    {train_df.shape}")
print(f"Test shape:     {test_df.shape}")
print(f"Original shape: {original_df.shape}")

TARGET = "exam_score"
ID_COL = "id"

base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = train_df.select_dtypes("object").columns.to_list()

print(f"\nBase features: {len(base_features)}")
print(f"Categorical features: {CATS}")

# ============================================================================
# 2. FEATURE ENGINEERING (ORIGINAL 46 FEATURES - WITH BINS)
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING (OPTIMIZED SET - 46 FEATURES)")
print("="*80)

def preprocess_optimized(df):
    """
    Generate high-value features INCLUDING binned features.
    Returns: (DataFrame with selected features, list of numeric feature names)
    """
    df_temp = df.copy()
    eps = 1e-5

    # Polynomials (2nd order only)
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    # Log transforms
    sh_pos = df_temp['study_hours'].clip(lower=0)
    ca_pos = df_temp['class_attendance'].clip(lower=0)
    sl_pos = df_temp['sleep_hours'].clip(lower=0)

    df_temp['log_study_hours'] = np.log1p(sh_pos)
    df_temp['log_class_attendance'] = np.log1p(ca_pos)
    df_temp['log_sleep_hours'] = np.log1p(sl_pos)

    # Sqrt transforms
    df_temp['sqrt_study_hours'] = np.sqrt(sh_pos)
    df_temp['sqrt_class_attendance'] = np.sqrt(ca_pos)

    # Key interactions
    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']
    df_temp['attendance_times_sleep'] = df_temp['class_attendance'] * df_temp['sleep_hours']
    df_temp['age_times_study_hours'] = df_temp['age'] * df_temp['study_hours']

    # Important ratios
    df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_sleep'] = df_temp['class_attendance'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_study'] = df_temp['class_attendance'] / (df_temp['study_hours'] + eps)

    # Ordinal encoding
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}

    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map(sleep_quality_map).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map(facility_rating_map).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map(exam_difficulty_map).fillna(1).astype(int)

    # Ordinal × numeric interactions
    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']

    # Ordinal × ordinal interactions
    df_temp['facility_x_sleepq'] = df_temp['facility_rating_numeric'] * df_temp['sleep_quality_numeric']
    df_temp['difficulty_x_facility'] = df_temp['exam_difficulty_numeric'] * df_temp['facility_rating_numeric']

    # Rule-based flags
    df_temp["high_att_high_study"] = ((df_temp["class_attendance"] >= 90) & (df_temp["study_hours"] >= 6)).astype(int)
    df_temp["ideal_sleep_flag"] = ((df_temp["sleep_hours"] >= 7) & (df_temp["sleep_hours"] <= 9)).astype(int)
    df_temp["high_study_flag"] = (df_temp["study_hours"] >= 7).astype(int)

    # Composite efficiency
    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)

    # Gap features
    df_temp['sleep_gap_8'] = (df_temp['sleep_hours'] - 8.0).abs()
    df_temp['attendance_gap_100'] = (df_temp['class_attendance'] - 100.0).abs()

    # BINNED FEATURES (KEEP THESE - THEY ARE VALUABLE!)
    df_temp['study_bin_num'] = pd.cut(df_temp['study_hours'], bins=5, labels=False).astype(int)
    df_temp['attendance_bin_num'] = pd.cut(df_temp['class_attendance'], bins=5, labels=False).astype(int)
    df_temp['sleep_bin_num'] = pd.cut(df_temp['sleep_hours'], bins=5, labels=False).astype(int)
    df_temp['age_bin_num'] = pd.cut(df_temp['age'], bins=5, labels=False).astype(int)

    # Feature list (34 features total)
    numeric_features = [
        'study_hours_squared', 'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
        'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
        'sqrt_study_hours', 'sqrt_class_attendance',
        'study_hours_times_attendance', 'study_hours_times_sleep', 'attendance_times_sleep',
        'age_times_study_hours',
        'study_hours_over_sleep', 'attendance_over_sleep', 'attendance_over_study',
        'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric',
        'study_hours_times_sleep_quality', 'attendance_times_facility', 'sleep_hours_times_difficulty',
        'facility_x_sleepq', 'difficulty_x_facility',
        'high_att_high_study', 'ideal_sleep_flag', 'high_study_flag',
        'efficiency',
        'sleep_gap_8', 'attendance_gap_100',
        'study_bin_num', 'attendance_bin_num', 'sleep_bin_num', 'age_bin_num'
    ]

    return df_temp[base_features + numeric_features], numeric_features

X_raw, numeric_cols = preprocess_optimized(train_df)
y = train_df[TARGET].reset_index(drop=True)

X_test_raw, _ = preprocess_optimized(test_df)
X_orig_raw, _ = preprocess_optimized(original_df)
y_orig = original_df[TARGET].reset_index(drop=True)

full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0, ignore_index=True)

for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

print(f"Engineered features: {len(numeric_cols)}")
print(f"Total features: {X.shape[1]} (11 base + {len(numeric_cols)} engineered)")

# ============================================================================
# 3. RIDGE REGRESSION META-FEATURE
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING RIDGE REGRESSION META-FEATURE")
print("="*80)

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(X.shape[0])
test_preds_lr = np.zeros((X_test.shape[0], FOLDS))
orig_preds_lr = np.zeros(X_original.shape[0])

for fold, (train_index, val_index) in enumerate(kf.split(X, y), start=1):
    X_train_fold, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    X_train_combined = pd.concat([X_train_fold, X_original], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    target_encoder = TargetEncoder(smooth='auto', target_type='continuous')
    X_train_encoded = X_train_combined.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()

    X_train_encoded[CATS] = target_encoder.fit_transform(X_train_combined[CATS], y_train_combined)
    X_val_encoded[CATS] = target_encoder.transform(X_val[CATS])
    X_test_encoded[CATS] = target_encoder.transform(X_test[CATS])

    alphas = np.logspace(-3, 3, 20)
    lr_model = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    lr_model.fit(X_train_encoded, y_train_combined.to_numpy().ravel())

    lr_val_pred = np.clip(lr_model.predict(X_val_encoded), 0, 100)
    lr_test_pred = np.clip(lr_model.predict(X_test_encoded), 0, 100)
    lr_orig_pred = np.clip(lr_model.predict(X_train_encoded.iloc[-X_original.shape[0]:]), 0, 100)

    oof_pred_lr[val_index] = lr_val_pred
    test_preds_lr[:, fold - 1] = lr_test_pred
    orig_preds_lr += lr_orig_pred / FOLDS

    rmse_lr = np.sqrt(mean_squared_error(y_val, lr_val_pred))
    print(f"Fold {fold:2d} | RMSE: {rmse_lr:.6f}")

lr_oof_rmse = np.sqrt(mean_squared_error(y, oof_pred_lr))
print(f"\nRidge OOF RMSE: {lr_oof_rmse:.6f}")

# ============================================================================
# 4. PREPARE DATASETS WITH RIDGE META-FEATURE
# ============================================================================

print(f"\n{'='*80}")
print("PREPARING XGBOOST DATASETS")
print("="*80)

# For XGBoost (categorical features as category type)
for col in base_features:
    full_data[col] = full_data[col].astype(str).astype("category")

for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X_xgb = full_data.iloc[:len(train_df)].copy()
X_test_xgb = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original_xgb = full_data.iloc[len(train_df) + len(test_df):].copy()

X_xgb["feature_lr_pred"] = oof_pred_lr
X_test_xgb["feature_lr_pred"] = test_preds_lr.mean(axis=1)
X_original_xgb["feature_lr_pred"] = orig_preds_lr

print(f"Final feature count: {X_xgb.shape[1]} (including Ridge meta-feature)")

# ============================================================================
# 5. XGBOOST TRAINING (OPTIMIZED HYPERPARAMETERS)
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING XGBOOST (OPTIMIZED)")
print("="*80)

xgb_params = {
    "n_estimators": 20000,           # More trees for better convergence
    "learning_rate": 0.004,          # Slightly slower (was 0.005)
    "max_depth": 9,                  # Keep same
    "subsample": 0.78,               # Slightly more data (was 0.75)
    "reg_lambda": 6,                 # More L2 regularization (was 5)
    "reg_alpha": 0.15,               # More L1 regularization (was 0.1)
    "colsample_bytree": 0.55,        # More features per tree (was 0.5)
    "colsample_bynode": 0.65,        # More features per node (was 0.6)
    "min_child_weight": 6,           # More conservative (was 5)
    "tree_method": "hist",
    "random_state": 42,
    "early_stopping_rounds": 100,    # More patience (was 80)
    "eval_metric": "rmse",
    "enable_categorical": True,
    "device": "cuda"
}

test_predictions_xgb = []
oof_predictions_xgb = np.zeros(len(X_xgb), dtype=float)

for fold, (train_index, val_index) in enumerate(kf.split(X_xgb, y), start=1):
    print(f"\nFold {fold:2d}/{FOLDS}")

    X_train_fold, X_val = X_xgb.iloc[train_index], X_xgb.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    X_train_combined = pd.concat([X_train_fold, X_original_xgb], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_combined, y_train_combined, eval_set=[(X_val, y_val)], verbose=1000)

    val_preds = model.predict(X_val)
    oof_predictions_xgb[val_index] = val_preds

    rmse_fold = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE: {rmse_fold:.5f}")

    test_predictions_xgb.append(model.predict(X_test_xgb))

xgb_oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions_xgb))

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("FINAL RESULTS")
print("="*80)

print(f"\nModel Performance:")
print(f"  Ridge OOF RMSE:    {lr_oof_rmse:.6f}")
print(f"  XGBoost OOF RMSE:  {xgb_oof_rmse:.5f}")

print(f"\nFeature Summary:")
print(f"  Base features:       {len(base_features)}")
print(f"  Engineered features: {len(numeric_cols)}")
print(f"  Meta-feature (Ridge): 1")
print(f"  Total features:      {X_xgb.shape[1]}")

# Save OOF predictions
oof_xgb = pd.DataFrame({"id": train_df[ID_COL], TARGET: oof_predictions_xgb})
oof_xgb.to_csv("xgb_oof_optimized.csv", index=False)

# Save submission
test_xgb_avg = np.mean(test_predictions_xgb, axis=0)
submission_xgb = submission_df.copy()
submission_xgb[TARGET] = test_xgb_avg
submission_xgb.to_csv("submission_optimized.csv", index=False)

print(f"\nFiles saved:")
print(f"  submission_optimized.csv (XGBoost - SUBMIT THIS)")
print(f"  xgb_oof_optimized.csv")

# ============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

importance_dict_xgb = model.get_booster().get_score(importance_type="gain")
feature_importance = pd.DataFrame({
    "feature": list(importance_dict_xgb.keys()),
    "importance": list(importance_dict_xgb.values())
}).sort_values("importance", ascending=False)

feature_importance['importance_pct'] = 100 * feature_importance['importance'] / feature_importance['importance'].sum()

feature_importance.to_csv("feature_importance_optimized.csv", index=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10)[["feature", "importance_pct"]].to_string(index=False))

print(f"\nSaved: feature_importance_optimized.csv")

print(f"\n{'='*80}")
print("TRAINING COMPLETE - SUBMIT: submission_optimized.csv")
print("="*80)
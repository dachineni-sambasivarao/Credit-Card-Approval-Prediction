"""
Credit Card Approval Prediction - Full Training Pipeline
=========================================================
Fixes the Random Forest deployment bug by saving a complete
Pipeline (preprocessing + model) instead of a bare RandomForestClassifier
that was trained on a separately-fitted, unsaved ColumnTransformer.
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("xgboost not installed - skipping XGBoost comparison (run `pip install xgboost` to include it)")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------
application_df = pd.read_csv("Dataset/application_record.csv")
credit_df = pd.read_csv("Dataset/credit_record.csv")

print("Application Dataset Shape:", application_df.shape)
print("Credit Dataset Shape:", credit_df.shape)


# ---------------------------------------------------------------------
# 2. Build target variable from credit history
#    STATUS codes 1-5 => risky/late payment history => TARGET = 1 (Reject)
#    Everything else  => TARGET = 0 (Approve)
# ---------------------------------------------------------------------
def create_target(statuses):
    if any(status in ["1", "2", "3", "4", "5"] for status in statuses):
        return 1  # Rejected (High Risk)
    return 0      # Approved (Low Risk)


target_df = (
    credit_df
    .groupby("ID")["STATUS"]
    .apply(list)
    .reset_index()
)
target_df["TARGET"] = target_df["STATUS"].apply(create_target)
target_df = target_df[["ID", "TARGET"]]


# ---------------------------------------------------------------------
# 3. Merge applications with target, clean data
# ---------------------------------------------------------------------
df = application_df.merge(target_df, on="ID", how="inner")

# Fill missing occupation
df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop ID (not a predictive feature)
if "ID" in df.columns:
    df.drop("ID", axis=1, inplace=True)

print("Final dataset shape:", df.shape)
print(df["TARGET"].value_counts())


# ---------------------------------------------------------------------
# 4. Split features / target
# ---------------------------------------------------------------------
X = df.drop("TARGET", axis=1)
y = df["TARGET"]

# These are the RAW columns the Flask app must supply, in this exact order
FEATURE_COLUMNS = X.columns.tolist()
print("\nFeature columns (raw, as expected by the saved pipeline):")
print(FEATURE_COLUMNS)

numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include="object").columns.tolist()

print("\nNumerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)


# ---------------------------------------------------------------------
# 5. Train/test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ---------------------------------------------------------------------
# 6. Preprocessing: scale numeric, one-hot encode categorical
#    IMPORTANT: this preprocessor is now bundled INTO the saved model,
#    so it no longer needs to be re-created/re-fitted at inference time.
# ---------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough",
)


# ---------------------------------------------------------------------
# 7. Helper to train + evaluate a model as part of a full Pipeline
# ---------------------------------------------------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("=" * 60)
    print(model.__class__.__name__)
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return pipeline, (accuracy, precision, recall, f1, roc_auc)


# ---------------------------------------------------------------------
# 8. Train candidate models (raw X_train/X_test now — preprocessing
#    happens INSIDE each pipeline, not as a separate manual step)
# ---------------------------------------------------------------------
log_pipeline, log_metrics = evaluate_model(
    LogisticRegression(random_state=42), X_train, X_test, y_train, y_test
)

dt_pipeline, dt_metrics = evaluate_model(
    DecisionTreeClassifier(random_state=42), X_train, X_test, y_train, y_test
)

rf_pipeline, rf_metrics = evaluate_model(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, X_test, y_train, y_test
)

if XGBOOST_AVAILABLE:
    xgb_pipeline, xgb_metrics = evaluate_model(
        XGBClassifier(random_state=42, eval_metric="logloss"),
        X_train, X_test, y_train, y_test
    )
else:
    xgb_pipeline, xgb_metrics = None, (None, None, None, None, None)


# ---------------------------------------------------------------------
# 9. Compare models
# ---------------------------------------------------------------------
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
    "Accuracy": [log_metrics[0], dt_metrics[0], rf_metrics[0], xgb_metrics[0]],
    "Precision": [log_metrics[1], dt_metrics[1], rf_metrics[1], xgb_metrics[1]],
    "Recall": [log_metrics[2], dt_metrics[2], rf_metrics[2], xgb_metrics[2]],
    "F1 Score": [log_metrics[3], dt_metrics[3], rf_metrics[3], xgb_metrics[3]],
    "ROC AUC": [log_metrics[4], dt_metrics[4], rf_metrics[4], xgb_metrics[4]],
})

print("\nModel comparison:")
print(results.sort_values(by="Accuracy", ascending=False))


# ---------------------------------------------------------------------
# 10. Save the FULL Random Forest pipeline (preprocessing + model)
#     This is the key fix: app.py can now feed it raw form values
#     directly, since the pipeline handles scaling + one-hot encoding.
# ---------------------------------------------------------------------
joblib.dump(rf_pipeline, "models/best_credit_card_model.pkl")
joblib.dump(FEATURE_COLUMNS, "models/feature_columns.pkl")

print("\nSaved models/best_credit_card_model.pkl (full pipeline)")
print("Saved models/feature_columns.pkl (raw column order Flask must use)")

# Sanity check: reload and confirm it accepts raw data
loaded = joblib.load("models/best_credit_card_model.pkl")
sample = X_test.iloc[[0]]
print("\nSanity check prediction on one raw test row:", loaded.predict(sample))

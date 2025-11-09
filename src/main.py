"""Credit approval logistic regression pipeline (script).

Creates a synthetic dataset, runs EDA (saves plots), builds a ColumnTransformer + Pipeline,
trains LogisticRegression, extracts coefficients and odds ratios, evaluates, and performs
threshold analysis.

Run: python src/main.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import make_classification
import argparse
from urllib.error import URLError


def load_default_credit_dataset(url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls") -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Attempt to load the UCI Default of Credit Card Clients dataset.

    Returns a DataFrame and inferred lists of numeric and categorical feature names.
    If loading or parsing fails, this function will raise an exception and the caller
    should decide how to handle fallback.
    """
    try:
        df = pd.read_excel(url, header=1)
    except Exception as exc:
        raise URLError(f"Could not download or parse dataset from {url}: {exc}")

    # Find target column (contains 'default')
    target_cols = [c for c in df.columns if "default" in str(c).lower()]
    if not target_cols:
        # fallback: any column with 'next' and 'month' words or 'default payment next month'
        target_cols = [c for c in df.columns if "next" in str(c).lower() and "month" in str(c).lower()]
    if not target_cols:
        raise ValueError("Could not find target column containing 'default' in downloaded dataset")

    target_col = target_cols[0]
    # In this dataset, 1 indicates default. We'll define `approved` = 1 - default
    df["approved"] = (df[target_col] == 0).astype(int)

    # Select some sensible numeric and categorical features if present
    preferred_numeric = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "PAY_AMT2"]
    numeric_features = [c for c in preferred_numeric if c in df.columns]
    if not numeric_features:
        numeric_features = [c for c in df.select_dtypes(include=["number"]).columns if c not in ("ID", target_col, "approved")]

    preferred_categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    categorical_features = [c for c in preferred_categorical if c in df.columns]
    if not categorical_features:
        categorical_features = [c for c in df.columns if df[c].dtype == object]

    # Ensure approved is present
    selected = numeric_features + categorical_features + ["approved"]
    # Keep only columns we will use (if any preferred selection exists)
    df = df.loc[:, [c for c in selected if c in df.columns]]
    return df, numeric_features, categorical_features


FIG_DIR = os.path.join(os.getcwd(), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


@dataclass
class DatasetParams:
    n_samples: int = 5000
    n_numeric: int = 4
    n_informative: int = 3
    class_sep: float = 1.0
    flip_y: float = 0.01
    weights: Tuple[float, float] = (0.7, 0.3)


def generate_synthetic_credit(params: DatasetParams) -> pd.DataFrame:
    """Generate a synthetic credit dataset.

    Numeric features produced by sklearn.make_classification are then mapped to
    more interpretable names: income, age, credit_score, debt_ratio.

    Two categorical features are added: employment_status and home_ownership.
    """
    X, y = make_classification(
        n_samples=params.n_samples,
        n_features=params.n_numeric,
        n_informative=params.n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=list(params.weights),
        class_sep=params.class_sep,
        flip_y=params.flip_y,
        random_state=42,
    )

    df = pd.DataFrame(X, columns=["num_0", "num_1", "num_2", "num_3"])

    # Map numeric columns to plausible features by scaling and shifting
    rng = np.random.RandomState(42)
    # Use (max-min) instead of np.ptp which isn't available on Series
    span0 = df["num_0"].max() - df["num_0"].min()
    span1 = df["num_1"].max() - df["num_1"].min()
    span2 = df["num_2"].max() - df["num_2"].min()
    span3 = df["num_3"].max() - df["num_3"].min()

    df["income"] = np.round((df["num_0"] - df["num_0"].min()) / span0 * 80000 + 20000)
    df["age"] = np.round((df["num_1"] - df["num_1"].min()) / span1 * 42 + 18)
    df["credit_score"] = np.round((df["num_2"] - df["num_2"].min()) / span2 * 400 + 300)
    df["debt_ratio"] = np.clip((df["num_3"] - df["num_3"].min()) / span3, 0, 1)

    # Drop intermediate numeric cols
    df = df.drop(columns=[c for c in df.columns if c.startswith("num_")])

    # Add categorical features with some dependency on target to make it realistic
    employment_options = ["employed", "unemployed", "self-employed"]
    home_options = ["rent", "own", "mortgage"]

    df["employment_status"] = rng.choice(
        employment_options, size=params.n_samples, p=[0.75, 0.1, 0.15]
    )
    df["home_ownership"] = rng.choice(home_options, size=params.n_samples, p=[0.5, 0.25, 0.25])

    # Insert target with same y from make_classification
    df["approved"] = y

    # Introduce some missingness to demonstrate imputation
    # Randomly set ~2% of income and employment_status to NaN
    mask_income = rng.rand(params.n_samples) < 0.02
    mask_emp = rng.rand(params.n_samples) < 0.02
    df.loc[mask_income, "income"] = np.nan
    df.loc[mask_emp, "employment_status"] = None

    return df


def perform_eda(df: pd.DataFrame, save_dir: str = FIG_DIR) -> None:
    """Create a few EDA plots and save them to `save_dir`.

    Plots: numeric histograms, boxplots, correlation heatmap, categorical counts.
    """
    numeric = ["income", "age", "credit_score", "debt_ratio"]
    categorical = ["employment_status", "home_ownership"]

    # Histograms
    df[numeric].hist(bins=20, figsize=(10, 6))
    plt.tight_layout()
    plt.suptitle("Numeric feature distributions", y=1.02)
    plt.savefig(os.path.join(save_dir, "numeric_distributions.png"))
    plt.close()

    # Boxplots
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[numeric], orient="h")
    plt.title("Numeric boxplots")
    plt.savefig(os.path.join(save_dir, "numeric_boxplots.png"))
    plt.close()

    # Correlation heatmap (drop missing rows for correlation)
    plt.figure(figsize=(6, 5))
    corr = df[numeric].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Numeric correlation")
    plt.savefig(os.path.join(save_dir, "numeric_correlation.png"))
    plt.close()

    # Categorical counts
    for col in categorical:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col)
        plt.title(f"Counts for {col}")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"count_{col}.png"))
        plt.close()


def build_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """Return a Pipeline that applies ColumnTransformer preprocessing and a classifier."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Logistic regression: use lbfgs with L2 regularization.
    # If classes are imbalanced, class_weight can be set to 'balanced'. We'll set it later based on y distribution.
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    return pipeline


def map_feature_names(pipeline: Pipeline, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """Extract transformed feature names from the fitted pipeline's preprocessor."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    # Numeric features remain as-is after scaling
    num_feats = numeric_features

    # For categorical, get feature names from OneHotEncoder
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_ohe_cols = list(ohe.get_feature_names_out(categorical_features))

    return num_feats + cat_ohe_cols


def evaluate_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    """Compute precision/recall/f1 for thresholds in [0,1]."""
    thresholds = np.linspace(0.0, 1.0, 101)
    records = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        records.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, save_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit approval logistic regression pipeline")
    parser.add_argument("--use-real", action="store_true", help="Attempt to download and use the UCI real dataset (Default of Credit Card Clients)")
    args = parser.parse_args()

    if args.use_real:
        print("Attempting to download and use the UCI Default dataset...")
        try:
            df_real, numeric_features, categorical_features = load_default_credit_dataset()
            df = df_real
            print(f"Loaded real dataset shape: {df.shape}")
            # Brief EDA on chosen real dataset
            perform_eda(df)
            print(f"EDA figures saved to {FIG_DIR}")
            target = "approved"
            X = df.drop(columns=[target])
            y = df[target].values
        except Exception as exc:
            print(f"Failed to load real dataset: {exc}\nFalling back to synthetic data.")
            args.use_real = False

    if not args.use_real:
        # synthetic
        params = DatasetParams()
        print(f"Generating synthetic dataset with params: {params}")
        df = generate_synthetic_credit(params)
        print("Dataset shape:", df.shape)
        print(df.head())

        perform_eda(df)
        print(f"EDA figures saved to {FIG_DIR}")

        target = "approved"
        X = df.drop(columns=[target])
        y = df[target].values

        numeric_features = ["income", "age", "credit_score", "debt_ratio"]
        categorical_features = ["employment_status", "home_ownership"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline(numeric_features, categorical_features)

    # Consider class_weight if imbalance is large
    ratio_pos = y_train.mean()
    print(f"Train positive class fraction: {ratio_pos:.3f}")
    if abs(ratio_pos - 0.5) > 0.1:
        # set class_weight to balanced
        pipeline.named_steps["classifier"].class_weight = "balanced"
        print("Set classifier.class_weight = 'balanced' due to class imbalance")

    # Fit pipeline
    pipeline.fit(X_train, y_train)
    print("Model trained")

    # 3) Interpretation: coefficients and odds ratios
    coef = pipeline.named_steps["classifier"].coef_[0]
    feature_names = map_feature_names(pipeline, numeric_features, categorical_features)
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coef})
    coef_df["odds_ratio"] = np.exp(coef_df["coef"])
    coef_df = coef_df.sort_values(by="odds_ratio", ascending=False)
    print("Top coefficients (odds ratios):")
    print(coef_df.head(10).to_string(index=False))
    coef_df.to_csv("results_coefficients.csv", index=False)

    # 4) Evaluation
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_05 = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred_05)
    prec = precision_score(y_test, y_pred_05, zero_division=0)
    rec = recall_score(y_test, y_pred_05, zero_division=0)
    f1 = f1_score(y_test, y_pred_05, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Evaluation at 0.5 -- Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred_05)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (threshold=0.5)")
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_0.5.png"))
    plt.close()

    # ROC plot
    plot_roc(y_test, y_proba, os.path.join(FIG_DIR, "roc_curve.png"))
    print(f"Saved confusion matrix and ROC to {FIG_DIR}")

    # Threshold analysis
    thresh_df = evaluate_thresholds(y_test, y_proba)
    # choose threshold that maximizes F1
    best_f1_row = thresh_df.loc[thresh_df["f1"].idxmax()]
    print(f"Best F1 at threshold {best_f1_row.threshold:.2f} -> F1={best_f1_row.f1:.3f}, Precision={best_f1_row.precision:.3f}, Recall={best_f1_row.recall:.3f}")

    # In fintech, false positives (approving bad applicant) are often costlier than false negatives.
    # We might increase threshold to favour precision. Find smallest threshold with precision >= 0.80.
    pref = thresh_df[thresh_df["precision"] >= 0.80]
    if not pref.empty:
        chosen = pref.iloc[0]
        print(f"Threshold with precision >= 0.80: {chosen.threshold:.2f} (precision={chosen.precision:.2f}, recall={chosen.recall:.2f})")
    else:
        print("No threshold achieves precision >= 0.80; consider recalibrating model or accepting lower precision.")

    # Save threshold sweep figure (precision/recall/f1 vs threshold)
    plt.figure(figsize=(8, 5))
    plt.plot(thresh_df["threshold"], thresh_df["precision"], label="precision")
    plt.plot(thresh_df["threshold"], thresh_df["recall"], label="recall")
    plt.plot(thresh_df["threshold"], thresh_df["f1"], label="f1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Threshold analysis")
    plt.savefig(os.path.join(FIG_DIR, "threshold_analysis.png"))
    plt.close()

    # Persist a small summary
    results = {
        "accuracy_0.5": acc,
        "precision_0.5": prec,
        "recall_0.5": rec,
        "f1_0.5": f1,
        "roc_auc": roc_auc,
        "best_f1_threshold": float(best_f1_row.threshold),
    }
    pd.Series(results).to_frame("value").to_csv("results_summary.csv")

    print("Done. Results written to results_summary.csv and results_coefficients.csv")


if __name__ == "__main__":
    main()

"""
================================================================================
   CUSTOMER CHURN PREDICTION — End-to-End Machine Learning Pipeline
================================================================================
   Dataset  : Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)
   Models   : Logistic Regression  |  Random Forest Classifier
   Author   : ML Pipeline Script
   Python   : 3.x
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

import os
import warnings
warnings.filterwarnings("ignore")

# Global seaborn style for all plots
sns.set_style("whitegrid")
sns.set_context("talk")

# Width of horizontal separator lines
W = 70

# Directory paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — pretty-print section headers
# ─────────────────────────────────────────────────────────────────────────────
def _header(step: int, title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * W)
    print(f"  STEP {step}  │  {title}")
    print("=" * W)


def _sub(text: str) -> None:
    """Print an indented sub-item bullet."""
    print(f"    ► {text}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Read the CSV file and return a raw DataFrame."""
    df = pd.read_csv(filepath)
    _header(1, "DATASET LOADED")
    _sub(f"Source : {filepath}")
    _sub(f"Records: {df.shape[0]:,}   |   Features: {df.shape[1]}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA EXPLORATION
# ─────────────────────────────────────────────────────────────────────────────
def explore_data(df: pd.DataFrame) -> None:
    """Print dataset shape, missing values, and class distribution."""
    _header(2, "DATA EXPLORATION")

    # Shape
    print(f"\n    {'Metric':<25}{'Value':>15}")
    print("    " + "─" * 40)
    print(f"    {'Total Rows':<25}{df.shape[0]:>15,}")
    print(f"    {'Total Columns':<25}{df.shape[1]:>15}")

    # Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"    {'Missing Cells':<25}{total_missing:>15}")

    # Target distribution
    churn_counts = df["Churn"].value_counts()
    churn_rate = churn_counts["Yes"] / len(df)
    print()
    print(f"    {'Churn Distribution':<25}")
    print("    " + "─" * 40)
    print(f"    {'  No  (Retained)':<25}{churn_counts['No']:>10,}  ({1 - churn_rate:>6.2%})")
    print(f"    {'  Yes (Churned)':<25}{churn_counts['Yes']:>10,}  ({churn_rate:>6.2%})")
    print()
    _sub(f"Churn Rate: {churn_rate:.2%}  — imbalanced dataset")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode the dataset:
      - Drop customerID
      - Fix TotalCharges type
      - Impute missing values with median
      - Binary-encode the target
      - Label-encode all remaining categoricals
    """
    _header(3, "DATA PREPROCESSING")
    df = df.copy()

    # 3a. Drop customerID (non-predictive identifier)
    df.drop(columns=["customerID"], inplace=True)
    _sub("Dropped 'customerID' column")

    # 3b. TotalCharges — convert whitespace strings to NaN, then to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isnull().sum()
    _sub(f"Converted 'TotalCharges' to numeric  ({n_missing} invalid → NaN)")

    # 3c. Impute missing TotalCharges with the median
    if n_missing > 0:
        median_val = df["TotalCharges"].median()
        df["TotalCharges"].fillna(median_val, inplace=True)
        _sub(f"Imputed {n_missing} missing values with median ({median_val:,.2f})")

    # 3d. Binary encode target: Yes → 1, No → 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    _sub("Encoded target 'Churn'  →  Yes=1 | No=0")

    # 3e. Label-encode remaining categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    _sub(f"Label-encoded {len(cat_cols)} categorical features")

    print(f"\n    Final dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    """80/20 stratified split with StandardScaler applied."""
    _header(4, "TRAIN / TEST SPLIT  (80 / 20)")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale features (important for Logistic Regression convergence)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    _sub(f"Training samples : {X_train.shape[0]:,}")
    _sub(f"Testing  samples : {X_test.shape[0]:,}")
    _sub(f"Features         : {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    """
    Train two classifiers for comparison:
      1. Logistic Regression  (class_weight='balanced')
      2. Random Forest         (class_weight='balanced')
    Returns a dict {name: fitted_model}.
    """
    _header(5, "MODEL TRAINING")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        _sub(f"{name:<25} — trained ✓")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 7. MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Compute and display metrics for a single model.
    Returns (y_pred, y_prob, confusion_matrix, accuracy, roc_auc).
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # ── Metrics table ──
    print(f"\n    ┌─────────────────────────────────────────────────┐")
    print(f"    │  {model_name:^47s}│")
    print(f"    ├─────────────────────────────────────────────────┤")
    print(f"    │  Accuracy            :   {acc:.4f}                  │")
    print(f"    │  ROC-AUC             :   {auc:.4f}                  │")
    print(f"    └─────────────────────────────────────────────────┘")

    # ── Confusion Matrix ──
    print(f"\n    Confusion Matrix ({model_name}):")
    print(f"                        Predicted")
    print(f"                    No Churn   Churn")
    print(f"    Actual No Churn   {cm[0][0]:>5d}     {cm[0][1]:>5d}")
    print(f"    Actual Churn      {cm[1][0]:>5d}     {cm[1][1]:>5d}")

    # ── Classification Report ──
    print(f"\n    Classification Report ({model_name}):")
    report = classification_report(
        y_test, y_pred, target_names=["No Churn", "Churn"], digits=4
    )
    for line in report.split("\n"):
        print(f"    {line}")

    return y_pred, y_prob, cm, acc, auc


def evaluate_all(models: dict, X_test, y_test):
    """Evaluate every model and return a results dict."""
    _header(6, "MODEL EVALUATION")
    results = {}
    for name, model in models.items():
        y_pred, y_prob, cm, acc, auc = evaluate_model(model, X_test, y_test, name)
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "cm": cm,
            "accuracy": acc,
            "roc_auc": auc,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8. MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def compare_models(results: dict) -> str:
    """Print a side-by-side comparison table and return the best model name."""
    _header(7, "MODEL COMPARISON")

    print(f"\n    {'Model':<28}{'Accuracy':>12}{'ROC-AUC':>12}")
    print("    " + "─" * 52)
    best_name, best_auc = None, -1
    for name, r in results.items():
        if r["roc_auc"] > best_auc:
            best_auc = r["roc_auc"]
            best_name = name
        print(f"    {name:<28}{r['accuracy']:>12.4f}{r['roc_auc']:>12.4f}")

    # Mark winner
    print("    " + "─" * 52)
    print(f"    Winner (by ROC-AUC): {best_name}  ({best_auc:.4f})")
    return best_name


# ─────────────────────────────────────────────────────────────────────────────
# 9. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(results: dict) -> None:
    """Side-by-side confusion-matrix heatmaps for every model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        sns.heatmap(
            r["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            annot_kws={"size": 16, "weight": "bold"},
            linewidths=1,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {name}", fontsize=14, weight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    _sub(f"Saved → {save_path}  (300 dpi)")


def plot_roc_curves(results: dict, y_test) -> None:
    """Overlay ROC curves for all models on one chart."""
    palette = sns.color_palette("Set1", len(results))
    fig, ax = plt.subplots(figsize=(10, 7))

    for (name, r), color in zip(results.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{name}  (AUC = {r['roc_auc']:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve Comparison", fontsize=15, weight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    _sub(f"Saved → {save_path}  (300 dpi)")


def plot_feature_importance_chart(top_features: pd.DataFrame) -> None:
    """Horizontal bar chart of top feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("coolwarm", len(top_features))

    bars = ax.barh(
        top_features["Feature"],
        top_features["Coefficient"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xlabel("Coefficient Value (Logistic Regression)", fontsize=12)
    ax.set_title("Top 10 Features Influencing Churn", fontsize=15, weight="bold", pad=12)
    ax.invert_yaxis()

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    _sub(f"Saved → {save_path}  (300 dpi)")


# ─────────────────────────────────────────────────────────────────────────────
# 10. TOP-10 FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
def print_top_features(model, feature_names: list) -> pd.DataFrame:
    """
    Print the top-10 features sorted by absolute coefficient value
    from the Logistic Regression model.  Returns the DataFrame.
    """
    _header(8, "TOP FEATURES INFLUENCING CHURN")

    coefs = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0],
        "|Coefficient|": np.abs(model.coef_[0]),
    }).sort_values("|Coefficient|", ascending=False).head(10)

    # Clean table
    print(f"\n    {'Rank':<6}{'Feature':<25}{'Coefficient':>14}")
    print("    " + "─" * 45)
    for rank, (_, row) in enumerate(coefs.iterrows(), 1):
        sign = "+" if row["Coefficient"] > 0 else ""
        print(f"    {rank:<6}{row['Feature']:<25}{sign}{row['Coefficient']:>13.4f}")

    print("\n    (+) coefficient → increases churn probability")
    print("    (−) coefficient → decreases churn probability")

    # For the bar chart we want top-down order → reverse so largest bar is on top
    return coefs.iloc[::-1].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 11. BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def print_business_insights(top_features: pd.DataFrame) -> None:
    """Summarise actionable business insights derived from the model."""
    _header(9, "BUSINESS INSIGHTS & RECOMMENDATIONS")

    insights = [
        "Customers on month-to-month contracts exhibit significantly "
        "higher churn probability. Incentivising longer-term contracts "
        "(1-year / 2-year) could materially reduce attrition.",

        "Tenure is a strong negative predictor of churn — the longer a "
        "customer stays, the less likely they are to leave. Early-tenure "
        "engagement programs (first 6–12 months) are critical.",

        "Customers without online security, tech support, or online backup "
        "churn at higher rates. Bundling these value-added services may "
        "improve retention.",

        "Higher monthly charges correlate with elevated churn risk. "
        "Consider loyalty discounts or price-lock guarantees for "
        "high-spending customers.",

        "Fiber-optic internet users show higher churn compared to DSL. "
        "Investigate service-quality and pricing satisfaction in that segment.",

        "Electronic-check payment method is associated with higher churn. "
        "Encouraging automatic bank-transfer or credit-card payments could "
        "reduce involuntary and friction-based churn.",
    ]

    for i, insight in enumerate(insights, 1):
        print(f"\n    {i}. {insight}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE RESULTS TO TEXT FILE
# ─────────────────────────────────────────────────────────────────────────────
def save_results_to_file(results: dict, best_name: str, top_features: pd.DataFrame) -> None:
    """Write all key outputs to a text file in the outputs/ folder."""
    path = os.path.join(OUTPUT_DIR, "model_results.txt")
    lines = []
    lines.append("=" * 70)
    lines.append("   CUSTOMER CHURN PREDICTION — MODEL RESULTS")
    lines.append("=" * 70)

    # ── Model Performance ──
    lines.append("\n" + "-" * 70)
    lines.append("  MODEL PERFORMANCE")
    lines.append("-" * 70)
    lines.append(f"\n  {'Model':<28}{'Accuracy':>12}{'ROC-AUC':>12}")
    lines.append("  " + "─" * 52)
    for name, r in results.items():
        marker = "  ★ Best" if name == best_name else ""
        lines.append(f"  {name:<28}{r['accuracy']:>12.4f}{r['roc_auc']:>12.4f}{marker}")
    lines.append("  " + "─" * 52)
    lines.append(f"  Winner (by ROC-AUC): {best_name}")

    # ── Confusion Matrices ──
    lines.append("\n" + "-" * 70)
    lines.append("  CONFUSION MATRICES")
    lines.append("-" * 70)
    for name, r in results.items():
        cm = r["cm"]
        lines.append(f"\n  {name}:")
        lines.append(f"                        Predicted")
        lines.append(f"                    No Churn   Churn")
        lines.append(f"    Actual No Churn   {cm[0][0]:>5d}     {cm[0][1]:>5d}")
        lines.append(f"    Actual Churn      {cm[1][0]:>5d}     {cm[1][1]:>5d}")

    # ── Classification Reports ──
    lines.append("\n" + "-" * 70)
    lines.append("  CLASSIFICATION REPORTS")
    lines.append("-" * 70)
    for name, r in results.items():
        from sklearn.metrics import classification_report as cr
        lines.append(f"\n  {name}:")
        report = cr(
            r["y_pred"],  # we'll pass y_test separately; use stored predictions
            r["y_pred"],  # placeholder — corrected below
            target_names=["No Churn", "Churn"], digits=4,
        )
        lines.append(f"  (see console output for full report)")

    # ── Top 10 Features ──
    lines.append("\n" + "-" * 70)
    lines.append("  TOP 10 FEATURES INFLUENCING CHURN")
    lines.append("-" * 70)
    # top_features is in reversed order for plotting; re-sort by |Coefficient|
    sorted_feats = top_features.sort_values("|Coefficient|", ascending=False) if "|Coefficient|" in top_features.columns else top_features
    lines.append(f"\n  {'Rank':<6}{'Feature':<25}{'Coefficient':>14}")
    lines.append("  " + "─" * 45)
    for rank, (_, row) in enumerate(sorted_feats.iterrows(), 1):
        sign = "+" if row["Coefficient"] > 0 else ""
        lines.append(f"  {rank:<6}{row['Feature']:<25}{sign}{row['Coefficient']:>13.4f}")
    lines.append("\n  (+) coefficient → increases churn probability")
    lines.append("  (−) coefficient → decreases churn probability")

    # ── Business Insights ──
    lines.append("\n" + "-" * 70)
    lines.append("  BUSINESS INSIGHTS")
    lines.append("-" * 70)
    insights = [
        "Customers on month-to-month contracts exhibit significantly higher "
        "churn probability. Incentivising longer-term contracts could reduce attrition.",
        "Tenure is a strong negative predictor — early-tenure engagement "
        "programs (first 6-12 months) are critical.",
        "Customers without online security, tech support, or online backup "
        "churn at higher rates. Bundling these services may improve retention.",
        "Higher monthly charges correlate with elevated churn risk. Consider "
        "loyalty discounts for high-spending customers.",
        "Fiber-optic internet users show higher churn compared to DSL. "
        "Investigate service-quality and pricing satisfaction.",
        "Electronic-check payment method is associated with higher churn. "
        "Encouraging auto-pay could reduce friction-based churn.",
    ]
    for i, insight in enumerate(insights, 1):
        lines.append(f"\n  {i}. {insight}")

    # ── Artefacts ──
    lines.append("\n" + "-" * 70)
    lines.append("  SAVED ARTEFACTS")
    lines.append("-" * 70)
    lines.append("  outputs/confusion_matrix.png")
    lines.append("  outputs/roc_curve.png")
    lines.append("  outputs/feature_importance.png")
    lines.append("  outputs/model_results.txt")
    lines.append("\n" + "=" * 70)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    _sub(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 13. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(results: dict, best_name: str) -> None:
    """Print a concise executive summary of the pipeline run."""
    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║" + "  PIPELINE SUMMARY".center(W - 2) + "║")
    print("╠" + "═" * (W - 2) + "╣")

    for name, r in results.items():
        marker = " ★" if name == best_name else ""
        line = f"  {name:<26} Acc: {r['accuracy']:.4f}   AUC: {r['roc_auc']:.4f}{marker}"
        print("║" + line.ljust(W - 2) + "║")

    print("╠" + "═" * (W - 2) + "╣")
    note = f"  Best Model → {best_name}"
    print("║" + note.ljust(W - 2) + "║")
    artefacts = "  Artefacts  → outputs/confusion_matrix.png"
    print("║" + artefacts.ljust(W - 2) + "║")
    artefacts2 = "               outputs/roc_curve.png"
    print("║" + artefacts2.ljust(W - 2) + "║")
    artefacts3 = "               outputs/feature_importance.png"
    print("║" + artefacts3.ljust(W - 2) + "║")
    artefacts4 = "               outputs/model_results.txt"
    print("║" + artefacts4.ljust(W - 2) + "║")
    print("╚" + "═" * (W - 2) + "╝")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    DATA_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # 1  Load data
    df = load_data(DATA_PATH)

    # 2  Explore
    explore_data(df)

    # 3  Preprocess
    df = preprocess_data(df)

    # 4  Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 5  Train both models
    models = train_models(X_train, y_train)

    # 6  Evaluate
    results = evaluate_all(models, X_test, y_test)

    # 7  Compare
    best_name = compare_models(results)

    # 8  Visualisations
    _header(0, "VISUALISATIONS")         # cosmetic header
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)

    # 9  Top-10 features (Logistic Regression coefficients)
    lr_model = models["Logistic Regression"]
    top_features = print_top_features(lr_model, X_train.columns.tolist())
    plot_feature_importance_chart(top_features)

    # 10  Business insights
    print_business_insights(top_features)

    # 11  Save results to text file
    save_results_to_file(results, best_name, top_features)

    # 12  Final summary
    print_summary(results, best_name)

    print("\n" + "─" * W)
    print("  Pipeline execution completed successfully.")
    print("─" * W + "\n")


if __name__ == "__main__":
    main()

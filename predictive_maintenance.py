"""
Predictive Maintenance for Manufacturing Equipment
AI4I 2020 Predictive Maintenance Dataset

Predicts equipment failure modes (TWF, HDF, PWF, OSF, RNF) from sensor
readings. Compares Logistic Regression, Random Forest, and Gradient Boosting
classifiers with recall-optimized thresholding for critical failure detection.
"""

import warnings
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FEATURE_COLS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_data(path="ai4i2020.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Overall failure rate: {df['Machine failure'].mean():.2%}")
    for mode in FAILURE_MODES:
        print(f"  {mode}: {df[mode].sum()} events ({df[mode].mean():.2%})")
    return df


def engineer_features(df):
    df = df.copy()
    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power_proxy"] = (
        df["Rotational speed [rpm]"] * df["Torque [Nm]"] * 2 * np.pi / 60
    )
    df["wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["rpm_per_torque"] = df["Rotational speed [rpm]"] / (df["Torque [Nm]"] + 1e-6)
    return df


def get_feature_matrix(df):
    feature_cols = FEATURE_COLS + [
        "temp_diff",
        "power_proxy",
        "wear_torque",
        "rpm_per_torque",
    ]
    X = df[feature_cols].values
    y = df["Machine failure"].values
    return X, y, feature_cols


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        X_tr = X_train_s if name == "Logistic Regression" else X_train
        X_te = X_test_s if name == "Logistic Regression" else X_test
        model.fit(X_tr, y_train)
        proba = model.predict_proba(X_te)[:, 1]
        pred_default = (proba >= 0.5).astype(int)

        # recall-optimized threshold (operations priority: catch failures)
        prec, rec, thr = precision_recall_curve(y_test, proba)
        f2 = (5 * prec * rec) / (4 * prec + rec + 1e-9)
        best_idx = int(np.argmax(f2[:-1])) if len(thr) else 0
        best_thr = thr[best_idx] if len(thr) else 0.5
        pred_tuned = (proba >= best_thr).astype(int)

        results.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_test, proba),
                "pr_auc": average_precision_score(y_test, proba),
                "recall_default": recall_score(y_test, pred_default),
                "precision_default": precision_score(y_test, pred_default),
                "f1_default": f1_score(y_test, pred_default),
                "threshold_tuned": float(best_thr),
                "recall_tuned": recall_score(y_test, pred_tuned),
                "precision_tuned": precision_score(y_test, pred_tuned),
                "f1_tuned": f1_score(y_test, pred_tuned),
            }
        )
        fitted[name] = (model, proba, pred_tuned)

        print(f"\n{name}")
        print(f"  ROC-AUC: {results[-1]['roc_auc']:.4f}")
        print(f"  PR-AUC:  {results[-1]['pr_auc']:.4f}")
        print(
            f"  @0.50 -> recall={results[-1]['recall_default']:.3f}, "
            f"precision={results[-1]['precision_default']:.3f}"
        )
        print(
            f"  @{best_thr:.3f} (F2-opt) -> recall={results[-1]['recall_tuned']:.3f}, "
            f"precision={results[-1]['precision_tuned']:.3f}"
        )

    return pd.DataFrame(results), fitted


def plot_feature_importance(model, feature_names, out="feature_importance.png"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(order)), importances[order][::-1])
    plt.yticks(range(len(order)), [feature_names[i] for i in order[::-1]])
    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_confusion(y_test, pred, name, out):
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"],
    )
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_pr_curves(fitted, y_test, out="pr_curves.png"):
    plt.figure(figsize=(7, 5))
    for name, (_, proba, _) in fitted.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — Binary Failure Detection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def per_mode_analysis(df, fitted_model, feature_names):
    df_fe = engineer_features(df)
    X = df_fe[feature_names].values
    rows = []
    for mode in FAILURE_MODES:
        y = df_fe[mode].values
        if y.sum() < 10:
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
        )
        m = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        rows.append(
            {
                "failure_mode": mode,
                "n_positive_train": int(y_tr.sum()),
                "pr_auc": average_precision_score(y_te, proba),
                "roc_auc": roc_auc_score(y_te, proba),
            }
        )
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Predictive Maintenance — AI4I 2020")
    print("=" * 60)

    df = load_data()
    df = engineer_features(df)
    X, y, feature_names = get_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(y_train):,} ({y_train.mean():.2%} failure)")
    print(f"Test:  {len(y_test):,} ({y_test.mean():.2%} failure)")

    print("\n" + "=" * 60)
    print("Binary Failure Classification (Machine failure)")
    print("=" * 60)
    results, fitted = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_names
    )

    results.to_csv("model_comparison.csv", index=False)
    print("\nSaved model_comparison.csv")

    best_rf, _, best_pred = fitted["Random Forest"]
    plot_feature_importance(best_rf, feature_names)
    plot_confusion(y_test, best_pred, "Random Forest (tuned)", "confusion_rf.png")
    plot_pr_curves(fitted, y_test)
    print("Saved feature_importance.png, confusion_rf.png, pr_curves.png")

    print("\n" + "=" * 60)
    print("Per-Failure-Mode Analysis")
    print("=" * 60)
    mode_results = per_mode_analysis(pd.read_csv("ai4i2020.csv"), best_rf, feature_names)
    print(mode_results.to_string(index=False))
    mode_results.to_csv("per_mode_results.csv", index=False)
    print("\nSaved per_mode_results.csv")


if __name__ == "__main__":
    main()

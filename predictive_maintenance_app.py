"""Interactive predictive-maintenance dashboard.

Run with::

    streamlit run predictive_maintenance_app.py

Trains the three classifiers on launch (cached), then offers two interaction
modes:

1. **Live prediction** — adjust sensor sliders to see real-time failure
   probability under each model, plus a what-if curve showing how the
   probability moves as you sweep one feature.
2. **Batch scoring** — upload a CSV of sensor readings (any subset of the
   AI4I columns) and download predictions.

Also displays model-comparison metrics, feature importance, and per-failure-
mode breakdown computed on a held-out test split.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from predictive_maintenance import (
    FAILURE_MODES,
    FEATURE_COLS,
    RANDOM_STATE,
    engineer_features,
    get_feature_matrix,
    load_data,
    per_mode_analysis,
    train_and_evaluate,
)

st.set_page_config(page_title="Predictive Maintenance", layout="wide", page_icon="🔧")


# ---------------------------------------------------------------------------
# Cached training
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Training classifiers (one-time, ~10 s)...")
def train_models(data_path: str = "ai4i2020.csv"):
    df = pd.read_csv(data_path)
    df_fe = engineer_features(df)
    X, y, feature_names = get_feature_matrix(df_fe)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    results, fitted = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)
    return {
        "df": df,
        "df_fe": df_fe,
        "feature_names": feature_names,
        "X_test": X_test,
        "y_test": y_test,
        "results": results,
        "fitted": fitted,
    }


@st.cache_resource(show_spinner="Computing per-failure-mode metrics...")
def compute_mode_results(data_path: str = "ai4i2020.csv"):
    df = pd.read_csv(data_path)
    df_fe = engineer_features(df)
    _, _, feature_names = get_feature_matrix(df_fe)
    return per_mode_analysis(df, None, feature_names)


def predict_one(reading: dict, fitted: dict, feature_names: list[str]):
    """Run all three classifiers on a single sensor reading."""
    type_code = {"L": 0, "M": 1, "H": 2}[reading["Type"]]
    air_t = reading["Air temperature [K]"]
    proc_t = reading["Process temperature [K]"]
    rpm = reading["Rotational speed [rpm]"]
    torque = reading["Torque [Nm]"]
    wear = reading["Tool wear [min]"]

    derived = {
        "temp_diff": proc_t - air_t,
        "power_proxy": rpm * torque * 2 * np.pi / 60,
        "wear_torque": wear * torque,
        "rpm_per_torque": rpm / (torque + 1e-6),
    }
    raw = {
        "Type": type_code,
        "Air temperature [K]": air_t,
        "Process temperature [K]": proc_t,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        **derived,
    }
    x = np.array([[raw[c] for c in feature_names]])
    out = {}
    for name, (model, _, _) in fitted.items():
        proba = float(model.predict_proba(x)[0, 1])
        out[name] = proba
    return out, raw


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


st.title("Predictive Maintenance — AI4I 2020")
st.caption("Score live sensor readings against three classifiers trained on 10,000 historical machine cycles.")

models = train_models()
df = models["df"]
df_fe = models["df_fe"]
feature_names = models["feature_names"]
fitted = models["fitted"]
results = models["results"]
X_test = models["X_test"]
y_test = models["y_test"]

# ---- Sidebar: live sensor input -------------------------------------------

st.sidebar.header("Live sensor readings")
st.sidebar.caption("Adjust sliders to score a single machine cycle.")

reading = {
    "Type": st.sidebar.selectbox("Product quality variant", ["L", "M", "H"], index=1,
                                  help="L = low-quality (60 % of mix), M = medium (30 %), H = high (10 %)."),
    "Air temperature [K]": st.sidebar.slider("Air temperature (K)", 295.0, 305.0,
                                              float(df["Air temperature [K]"].mean()), 0.1),
    "Process temperature [K]": st.sidebar.slider("Process temperature (K)", 305.0, 315.0,
                                                  float(df["Process temperature [K]"].mean()), 0.1),
    "Rotational speed [rpm]": st.sidebar.slider("Rotational speed (rpm)", 1100, 2900,
                                                  int(df["Rotational speed [rpm]"].mean()), 5),
    "Torque [Nm]": st.sidebar.slider("Torque (Nm)", 3.0, 80.0,
                                      float(df["Torque [Nm]"].mean()), 0.5),
    "Tool wear [min]": st.sidebar.slider("Tool wear (min)", 0, 250,
                                          int(df["Tool wear [min]"].mean()), 1),
}

probs, raw = predict_one(reading, fitted, feature_names)

threshold = st.sidebar.slider("Failure-flag threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Below this probability the cycle is treated as healthy.")

# ---- Main panel: live prediction header -----------------------------------

best_model = max(probs, key=probs.get)
best_p = probs[best_model]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Logistic Regression", f"{probs['Logistic Regression']:.1%}")
c2.metric("Random Forest", f"{probs['Random Forest']:.1%}")
c3.metric("Gradient Boosting", f"{probs['Gradient Boosting']:.1%}")
c4.metric("Highest", f"{best_p:.1%}", delta=best_model, delta_color="off")

if best_p >= threshold:
    st.error(f"⚠️  **FAILURE LIKELY** — {best_model} flags failure at p = {best_p:.1%} (threshold {threshold:.0%}). "
             "Recommend preventive maintenance before next cycle.")
else:
    st.success(f"✅  **HEALTHY** — highest classifier reports p = {best_p:.1%} < {threshold:.0%}. Continue operation.")


# ---- Tabbed analytics -----------------------------------------------------

tab_what, tab_perf, tab_modes, tab_batch = st.tabs(
    ["What-if sweep", "Model performance", "Per-failure-mode", "Batch scoring"]
)


# What-if: sweep one feature, hold others fixed
with tab_what:
    st.markdown("Fix the other sensors at the values from the sidebar; sweep one feature across its plausible range to see how each model's failure probability responds.")
    sweep_feature = st.selectbox(
        "Feature to sweep",
        ["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]",
         "Process temperature [K]", "Air temperature [K]"]
    )
    grid_min, grid_max = float(df[sweep_feature].min()), float(df[sweep_feature].max())
    grid = np.linspace(grid_min, grid_max, 80)
    sweep_rows = []
    for v in grid:
        modified = dict(reading); modified[sweep_feature] = float(v)
        p, _ = predict_one(modified, fitted, feature_names)
        sweep_rows.append({"x": v, **p})
    sweep_df = pd.DataFrame(sweep_rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    for col in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
        ax.plot(sweep_df["x"], sweep_df[col], label=col, linewidth=1.6)
    ax.axvline(reading[sweep_feature], color="gray", linestyle="--", alpha=0.6,
               label="current setting")
    ax.axhline(threshold, color="red", linestyle=":", alpha=0.6, label=f"threshold {threshold:.0%}")
    ax.set_xlabel(sweep_feature); ax.set_ylabel("P(failure)")
    ax.set_title(f"Sensitivity: P(failure) vs. {sweep_feature}")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)


with tab_perf:
    st.markdown("Held-out test split: 2,500 rows, stratified.")
    show_results = results.copy()
    for col in ["roc_auc", "pr_auc", "recall_default", "precision_default", "f1_default",
                "recall_tuned", "precision_tuned", "f1_tuned"]:
        show_results[col] = show_results[col].map("{:.3f}".format)
    show_results["threshold_tuned"] = show_results["threshold_tuned"].map("{:.3f}".format)
    st.dataframe(show_results, hide_index=True, use_container_width=True)

    col_pr, col_cm = st.columns(2)
    with col_pr:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, (_, proba, _) in fitted.items():
            prec, rec, _ = precision_recall_curve(y_test, proba)
            ap = average_precision_score(y_test, proba)
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall")
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)
    with col_cm:
        chosen_name = st.selectbox("Confusion matrix for", list(fitted.keys()), index=2)
        _, _, pred = fitted[chosen_name]
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No failure", "Failure"], yticklabels=["No failure", "Failure"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{chosen_name} (F2-tuned)")
        st.pyplot(fig)

    st.markdown("**Feature importance — Gradient Boosting**")
    gb = fitted["Gradient Boosting"][0]
    imp = pd.Series(gb.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp.index, imp.values, color="#4c72b0")
    ax.set_xlabel("importance"); st.pyplot(fig)


with tab_modes:
    st.markdown("A separate Random Forest is trained per failure mode against the same features. Modes with PR-AUC near 1.0 are deterministic functions of the sensors; modes near 0 carry no predictable signal.")
    mode_df = compute_mode_results()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    mode_df_sorted = mode_df.sort_values("pr_auc")
    colors = ["#c44e52" if v < 0.5 else "#dd8452" if v < 0.9 else "#55a868" for v in mode_df_sorted["pr_auc"]]
    ax.barh(mode_df_sorted["failure_mode"], mode_df_sorted["pr_auc"], color=colors)
    for i, v in enumerate(mode_df_sorted["pr_auc"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_xlabel("PR-AUC"); ax.set_title("Predictability per failure mode")
    st.pyplot(fig)
    st.dataframe(mode_df, hide_index=True, use_container_width=True)


with tab_batch:
    st.markdown(
        "Upload a CSV with at least these columns: `Type`, `Air temperature [K]`, `Process temperature [K]`, "
        "`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`. Predictions for all three models are "
        "added as new columns and made available for download."
    )
    upload = st.file_uploader("Sensor readings CSV", type=["csv"])
    if upload is not None:
        try:
            batch = pd.read_csv(upload)
            batch_fe = engineer_features(batch)
            X_batch, _, _ = get_feature_matrix(batch_fe.assign(**{
                m: 0 for m in FAILURE_MODES + ["Machine failure"] if m not in batch_fe.columns
            }))
            preds = batch.copy()
            for name, (model, _, _) in fitted.items():
                preds[f"{name} P(failure)"] = model.predict_proba(X_batch)[:, 1]
            preds["Highest model P"] = preds[[c for c in preds if c.endswith("P(failure)")]].max(axis=1)
            preds["Flag (>= threshold)"] = preds["Highest model P"] >= threshold
            st.dataframe(preds.head(50), use_container_width=True)
            buf = io.BytesIO(); preds.to_csv(buf, index=False); buf.seek(0)
            st.download_button("Download predictions CSV", buf,
                               file_name="predictions.csv", mime="text/csv")
            n_flag = int(preds["Flag (>= threshold)"].sum())
            st.info(f"{n_flag} of {len(preds)} cycles ({n_flag/len(preds):.1%}) flagged for maintenance at threshold {threshold:.0%}.")
        except Exception as e:
            st.error(f"Could not score that file: {e}")
    else:
        st.caption("Or download the bundled `ai4i2020.csv` and re-upload to see batch scoring in action.")

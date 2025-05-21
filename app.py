import streamlit as st
import pandas as pd
import joblib
import yaml
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from src.generate_features import generate_features

st.set_page_config(layout="wide")

# Load available model versions
model_root = "runs"
versions = sorted([v for v in os.listdir(model_root) if v.isdigit()])

st.title("🌤️ Cloud Classification Prediction App")

selected_version = st.selectbox("Choose model version:", versions)

# Input features sidebar
st.sidebar.header("Input Features")

# Predict trigger button at the top of sidebar
trigger_predict = st.sidebar.button("🔍 Predict")

default_inputs = {
    "visible_mean": 80.0,
    "visible_max": 60.0,
    "visible_min": 40.0,
    "visible_mean_distribution": 2.05,
    "visible_contrast": 10.0,
    "visible_entropy": 0.03,
    "visible_second_angular_momentum": 2.5,
    "IR_mean": 10.0,
    "IR_max": 120.0,
    "IR_min": 0.0
}

example_options = {
    "Cumulus (0)": {
        "visible_mean": 140.0,
        "visible_max": 85.0,
        "visible_min": 35.0,
        "visible_mean_distribution": 0.03,
        "visible_contrast": 500.0,
        "visible_entropy": 0.02,
        "visible_second_angular_momentum": 2.8,
        "IR_mean": 155.0,
        "IR_max": 230.0,
        "IR_min": 200.0
    },
    "Cirrus (1)": {
        "visible_mean": 16.9689,
        "visible_max": 58.6755,
        "visible_min": 40.9075,
        "visible_mean_distribution": 3.0208,
        "visible_contrast": 18.2875,
        "visible_entropy": 0.1375,
        "visible_second_angular_momentum": 2.1754,
        "IR_mean": -60.6180,
        "IR_max": 20.7812,
        "IR_min": -16.1559
    }
}

selected_example = st.sidebar.selectbox("🔍 Load Example Input", ["Custom Input"] + list(example_options.keys()))

user_input = {}
user_input_defaults = example_options.get(selected_example, default_inputs)
for key in default_inputs:
    user_input[key] = st.sidebar.number_input(key, value=user_input_defaults[key])

# Load config
def load_yaml(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"❌ Failed to load {path}: {e}")
        st.stop()

config_path = os.path.join(model_root, selected_version, "config.yaml")
config = load_yaml(config_path)

# Load model
try:
    model_path = os.path.join(model_root, selected_version, "trained_model_object.pkl")
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Run prediction if triggered
if trigger_predict:
    input_df = pd.DataFrame([user_input])
    try:
        processed_df = generate_features(input_df, config["generate_features"])
        features = config["train_model"]["initial_features"]
        X = processed_df[features]
        prediction = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        proba = probs[prediction]

        class_labels = {0: "Cumulus", 1: "Cirrus"}
        predicted_label = class_labels.get(prediction, str(prediction))

        st.success(f"✅ Predicted class: {predicted_label} ({prediction} | Probability: {proba * 100:.2f}%)")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


# Metrics display
metrics_path = os.path.join(model_root, selected_version, "metrics.yaml")
if os.path.exists(metrics_path):
    metrics = load_yaml(metrics_path)
    st.markdown("## 📊 Model Evaluation Metrics")

    if "classification_report" in metrics:
        report = metrics["classification_report"]

        st.markdown("### 🔍 Class-wise Performance")
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis('tight')
        ax.axis('off')
        table_data = [
            ["Class", "Precision", "Recall", "F1-Score"],
            ["Cumulus (0)", f"{report['0']['precision']:.2f}", f"{report['0']['recall']:.2f}", f"{report['0']['f1-score']:.2f}"],
            ["Cirrus (1)", f"{report['1']['precision']:.2f}", f"{report['1']['recall']:.2f}", f"{report['1']['f1-score']:.2f}"]
        ]
        table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 1.5)
        st.pyplot(fig)

        st.markdown("### 📌 Overall Metrics")
        overall = report.get("weighted avg", {})
        st.metric("F1-Score", f"{overall.get('f1-score', 0):.2f}")
        st.metric("Precision", f"{overall.get('precision', 0):.2f}")
        st.metric("Recall", f"{overall.get('recall', 0):.2f}")

    for k in ["accuracy", "auc"]:
        if k in metrics:
            st.metric(label=k.upper(), value=f"{metrics[k]:.4f}")

    st.markdown("### 🧮 ROC AUC Curve")
    score_path = os.path.join(model_root, selected_version, "scores.csv")
    if os.path.exists(score_path):
        try:
            score_df = pd.read_csv(score_path)
            fpr, tpr, _ = roc_curve(score_df["y_true"], score_df["y_score"])
            roc_auc = auc(fpr, tpr)

            pos_rate = score_df["y_true"].mean()
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"Model AUC = {roc_auc:.2f}")
            ax.plot([0, pos_rate, 1], [0, pos_rate, 1], color="gray", lw=2, linestyle="--", label=f"Random Classifier ({pos_rate:.2f})")
            ax.plot([0, 0, 1], [0, 1, 1], color="purple", lw=2, linestyle="-", label="Perfect Classifier")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ Failed to plot ROC curve: {e}")
    else:
        st.info("score.csv not found — ROC curve cannot be displayed.")

# Show visualizations
fig_dir = os.path.join(model_root, selected_version, "figures")
if os.path.exists(fig_dir):
    st.subheader("📈 Feature Distributions")
    fig_cols = st.columns(3)
    images = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
    for i, img in enumerate(sorted(images)):
        with fig_cols[i % 3]:
            st.image(os.path.join(fig_dir, img), caption=img, use_column_width=True)

# Data Source Reference
st.markdown("---")
st.markdown("ℹ️ **Data Source**: [UCI Cloud Data](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/taylor/cloud.data)")

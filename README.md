# ☁️ Cloud Classification Prediction App

This Streamlit app allows users to classify cloud types (Cumulus or Cirrus) using multiple trained machine learning model versions. It provides an interactive interface for model selection, prediction, metric visualization, and ROC curve analysis.

---

## 🔗 Live Web App

👉 [https://uuwzbtvt8tkx3jvunkxu7o.streamlit.app](https://uuwzbtvt8tkx3jvunkxu7o.streamlit.app)

---

## 📦 Repository

👉 [GitHub Repository](https://github.com/JunyiC8/cloud_hw3/blob/main/app.py)

---

## 🚀 How to Use the App

1. **Choose Model Version**  
   Use the dropdown menu at the top to select from available model versions (e.g., `1747717845`, `1747717830`).

2. **Input Features**  
   In the sidebar, either:
   - Load a preset sample (Cumulus or Cirrus), or  
   - Enter your own custom values for each feature.

3. **Click 🔍 Predict**  
   After providing input, click the "Predict" button at the top. The app will:
   - Show the predicted cloud class and its probability
   - Display model evaluation metrics (accuracy, AUC, precision, recall, F1)
   - Plot the ROC AUC Curve

4. **Switch Models Anytime**  
   You can change the model version without restarting the app.

---

## 📂 Project Structure

```
├── app.py                      # Main Streamlit app
├── runs/
│   ├── 1747717830/
│   │   ├── config.yaml
│   │   ├── metrics.yaml
│   │   ├── scores.csv
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── trained_model_object.pkl
│   │   └── figures/
│   └── 1747717845/
│       └── ...
├── src/
│   ├── generate_features.py
├── requirements.txt
└── README.md
```

---

## 📋 Dependencies

Install with:
```bash
pip install -r requirements.txt
```

Key libraries:
- `streamlit`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `pyyaml`
- `joblib`

---

## 📈 Data Source

Cloud classification dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/taylor/cloud.data)

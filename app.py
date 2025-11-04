
# app.py - Streamlit UI for Malware Detection (loads models from models/)
import streamlit as st
import pandas as pd, numpy as np, joblib, os
from pathlib import Path

st.set_page_config(page_title="Malware Detection Model", layout="wide")
st.title("Malware Detection Model (MLOps Demo)")

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/content/drive/MyDrive/MLOPS_Project/models"))
if not MODELS_DIR.exists():
    st.error("Models folder not found: " + str(MODELS_DIR))
    st.stop()

st.write("Found models at:", MODELS_DIR)

# Load metadata if exists
cols_info = None
try:
    cols_info = joblib.load(MODELS_DIR / 'cols_info.joblib')
    feature_columns = cols_info.get('feature_columns', [])
except Exception:
    feature_columns = []

# list model files
model_files = sorted([p.name for p in MODELS_DIR.glob('*_model.pkl')])
if not model_files:
    st.warning("No model files found in models directory.")
model_choice = st.selectbox("Select model", model_files)

uploaded = st.file_uploader("Upload CSV for batch prediction", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Uploaded shape:", df.shape)
    # Align with saved features if possible
    X = df.copy()
    if feature_columns:
        X = X.reindex(columns=feature_columns).fillna(0.0)
    else:
        X = X.fillna(0.0)

    # load scaler if exists
    scaler = None
    try:
        scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    except Exception:
        scaler = None

    X_vals = X.values
    if scaler is not None:
        try:
            X_vals = scaler.transform(X_vals)
        except Exception:
            pass

    model = joblib.load(MODELS_DIR / model_choice)
    preds = model.predict(X_vals)
    res = df.copy().reset_index(drop=True)
    res['prediction'] = preds
    st.dataframe(res.head(10))
    st.download_button("Download predictions CSV", res.to_csv(index=False).encode('utf-8'), file_name='predictions.csv')
else:
    st.info("Upload CSV for batch predictions or run training pipeline to create model files.")

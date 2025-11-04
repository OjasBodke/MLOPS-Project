
import streamlit as st
import pandas as pd, numpy as np, joblib, os
from pathlib import Path

st.set_page_config(page_title="Malware Detection Model", layout="wide")
st.title("Malware Detection (MLOps Demo)")

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/content/drive/MyDrive/MLOPS_Project/models"))
if not MODELS_DIR.exists():
    st.error("Models folder not found: " + str(MODELS_DIR))
    st.stop()

st.write("Models directory:", MODELS_DIR)

# Try to load metadata
feature_columns = None
try:
    cols_info = joblib.load(MODELS_DIR / 'cols_info.joblib')
    feature_columns = cols_info.get('feature_columns', None)
except Exception:
    feature_columns = None

model_files = sorted([p.name for p in MODELS_DIR.glob('*_model.pkl')])
model_choice = st.selectbox("Select model (loaded from models/)", model_files)

st.sidebar.header("Input")
input_mode = st.sidebar.radio("Mode", ["Manual (top features)", "Batch CSV"])

if input_mode == "Manual (top features)":
    st.header("Single Prediction (manual input)")
    # show top 5 features if available, else ask for first 5 numeric columns
    if feature_columns:
        top_feats = feature_columns[:5]
    else:
        top_feats = [f'F_{i}' for i in range(1,6)]
    vals = []
    for f in top_feats:
        v = st.number_input(f, value=0.0)
        vals.append(v)
    if st.button("Predict (single)"):
        model = joblib.load(MODELS_DIR / model_choice)
        scaler = None
        try:
            scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        except Exception:
            scaler = None
        X = np.array(vals).reshape(1, -1)
        if scaler is not None:
            X = scaler.transform(X)
        pred = model.predict(X)[0]
        st.success(f"Prediction: {pred}")

else:
    st.header("Batch Prediction (CSV upload)")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Uploaded data shape:", df.shape)
        # Align features
        if feature_columns:
            X = df.reindex(columns=feature_columns).fillna(0.0)
        else:
            X = df.fillna(0.0)
        model = joblib.load(MODELS_DIR / model_choice)
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
        preds = model.predict(X_vals)
        out = df.copy().reset_index(drop=True)
        out['prediction'] = preds
        st.dataframe(out.head(20))
        st.download_button("Download predictions CSV", out.to_csv(index=False).encode('utf-8'))

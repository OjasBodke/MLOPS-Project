
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle, os
import plotly.express as px
import time

# 🌌 Enhanced Dark Neon Theme
st.markdown('''
    <style>
        body {
            background: linear-gradient(to bottom right, #000000, #0d1117, #1a1a2e);
            color: #e0e0e0;
        }
        .stApp {
            background-color: transparent !important;
        }
        /* Title styling - black text */
        h1 {
            color: #000000 !important;
            text-shadow: 0px 0px 15px #00eaff;
        }
        h2, h3 {
            color: #ffcc00 !important;
        }
        /* Sidebar - Dark Cyber Look with black heading */
        div[data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #0d1117, #1e293b);
            color: #e0e0e0;
            border-right: 2px solid #00eaff;
            box-shadow: 2px 0px 15px #00eaff50;
        }
        div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {
            color: #000000 !important;
            text-shadow: none !important;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #00eaff, #0077b6);
            color: white;
            border-radius: 10px;
            border: none;
            transition: 0.3s;
            box-shadow: 0px 0px 10px #00eaff80;
        }
        div.stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #0096c7, #48cae4);
            box-shadow: 0px 0px 15px #00eaff;
        }
        .metric-label {color: #ffcc00 !important;}
    </style>
''', unsafe_allow_html=True)

# 🧠 Header (Black)
st.markdown("<h1 style='text-align:center;'>⚡ MLOps Malware Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

MODELS_DIR = Path("/content/drive/MyDrive/MLOPS_Project/models")
SCALER_PATH = Path("/content/drive/MyDrive/MLOPS_Project/models/scaler.pkl")
PERF_PATH = Path("/content/drive/MyDrive/MLOPS_Project/models/model_performance.csv")

# Sidebar (Black Heading)
st.sidebar.markdown("<h2 style='color:black;'>🎛️ Controls Panel</h2>", unsafe_allow_html=True)
selected_model_file = st.sidebar.selectbox("Choose a Model", sorted(os.listdir(MODELS_DIR)))
generate_sample = st.sidebar.button("✨ Generate Random Sample (5 rows)")
st.sidebar.info("Upload your dataset below or generate a sample to test predictions.")

# Leaderboard
if PERF_PATH.exists():
    perf_df = pd.read_csv(PERF_PATH)
    st.subheader("🏆 Model Accuracy Leaderboard")
    st.dataframe(perf_df.style.background_gradient(cmap='Blues'))
    fig = px.bar(perf_df, x='model', y='accuracy', title='Model Accuracy Comparison',
                 color='accuracy', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

# Load model & scaler
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(MODELS_DIR / selected_model_file, 'rb') as f:
    model = pickle.load(f)

st.markdown("---")
st.header("📤 Upload CSV for Batch Prediction")

uploaded = st.file_uploader("Upload CSV (only numeric features will be used)", type=['csv'])
df_input = None

# Sample generator
if generate_sample:
    try:
        n_features = scaler.mean_.shape[0]
    except:
        n_features = 10
    cols = [f"F_{i+1}" for i in range(n_features)]
    sample = pd.DataFrame(np.random.randint(0,2,size=(5,n_features)), columns=cols)
    st.success("✅ Sample data generated!")
    st.dataframe(sample)
    df_input = sample

if uploaded is not None:
    df_uploaded = pd.read_csv(uploaded)
    st.write("### Uploaded Data Preview")
    st.dataframe(df_uploaded.head())
    df_input = df_uploaded

# Prediction
if df_input is not None:
    st.markdown("---")
    with st.spinner("🚀 Running Predictions... Please wait"):
        time.sleep(1)
        df_num = df_input.select_dtypes(include=['number']).fillna(0)
        try:
            required_n = scaler.mean_.shape[0]
            if df_num.shape[1] < required_n:
                for i in range(df_num.shape[1], required_n):
                    df_num[f"PAD_{i}"] = 0
            elif df_num.shape[1] > required_n:
                df_num = df_num.iloc[:, :required_n]
        except Exception:
            st.warning("Couldn't detect required feature count; predictions may fail.")

        X_scaled = scaler.transform(df_num.values)
        preds = model.predict(X_scaled)
        df_out = df_input.copy()
        df_out['Prediction'] = preds
        df_out['Prediction_Label'] = df_out['Prediction'].map({0: '🟢 Non-Malicious', 1: '🔴 Malicious'})
        st.success("✅ Prediction complete!")
        st.write("### Results (first 50 rows)")
        st.dataframe(df_out.head(50))

        # 🎨 Light-colored pie chart
        counts = df_out['Prediction_Label'].value_counts().reset_index()
        counts.columns = ['Prediction_Label', 'Count']
        fig_pred = px.pie(counts, names='Prediction_Label', values='Count',
                          title="Prediction Distribution",
                          color_discrete_sequence=px.colors.qualitative.Pastel1)
        st.plotly_chart(fig_pred, use_container_width=True)

        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")

# ===============================================
# ⚡ MLOps Project — Machine Learning Model Deployment
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------
# 📁 MODEL DIRECTORY
# ------------------------------------------------
MODEL_DIR = "models"

# Automatically list available model files
available_models = [m for m in os.listdir(MODEL_DIR) if m.endswith(".pkl")]

# ------------------------------------------------
# 🎯 STREAMLIT APP UI
# ------------------------------------------------
st.set_page_config(page_title="MLOps Model Deployment", layout="centered")

st.title("🤖 MLOps Project — Model Deployment Dashboard")
st.markdown("---")
st.write("Welcome! Upload a CSV file and select a trained model to make predictions.")

# ------------------------------------------------
# 📤 FILE UPLOAD
# ------------------------------------------------
uploaded_file = st.file_uploader("Upload your CSV file for prediction", type=["csv"])

# ------------------------------------------------
# 🧠 MODEL SELECTION
# ------------------------------------------------
if available_models:
    selected_model_file = st.selectbox("Select a model", available_models)
    model_path = os.path.join(MODEL_DIR, selected_model_file)

    # Load model safely
    try:
        model = joblib.load(model_path)
        st.success(f"✅ Model '{selected_model_file}' loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
else:
    st.warning("⚠️ No models found in 'models/' directory.")
    model = None

# ------------------------------------------------
# 🔍 PREDICTION SECTION
# ------------------------------------------------
if uploaded_file is not None and model is not None:
    try:
        input_data = pd.read_csv(uploaded_file)
        st.write("### 🧾 Uploaded Data Preview")
        st.dataframe(input_data.head())

        # Drop 'Label' or any target columns automatically
        if 'Label' in input_data.columns:
            input_data = input_data.drop(columns=['Label'])
            st.info("ℹ️ 'Label' column detected and removed before prediction.")

        # Keep only numeric columns
        numeric_data = input_data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] < input_data.shape[1]:
            st.warning("⚠️ Non-numeric columns were removed before prediction.")

        if st.button("🚀 Predict"):
            try:
                predictions = model.predict(numeric_data)
                st.success("✅ Predictions generated successfully!")

                # Display predictions
                st.write("### 📊 Prediction Results")
                st.dataframe(pd.DataFrame(predictions, columns=["Predicted Output"]))

                # Download option
                output_df = input_data.copy()
                output_df["Predicted Output"] = predictions
                csv = output_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"Error reading file: {e}")

# ------------------------------------------------
# 🧾 FOOTER
# ------------------------------------------------
st.markdown("---")
st.markdown("Developed By Ojas Bodke")


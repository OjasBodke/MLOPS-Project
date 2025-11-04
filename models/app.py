
import streamlit as st
import pandas as pd, numpy as np, joblib, pickle
from pathlib import Path
from io import BytesIO

st.set_page_config(page_title="Malware Models Ensemble", layout="wide")
st.title("Malware Models Batch Prediction (Upload CSV)")

# Adjust this path if needed. When running locally, files are relative.
MODELS_DIR = Path("models")  # will be overwritten below in Colab usage

# Allow user to upload CSV
uploaded_file = st.file_uploader("Upload a CSV for batch prediction (features only or include the original target column)", type=["csv"])
st.write("Tip: The app will try to infer which columns are features using saved training metadata.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded shape:", df.shape)
    st.dataframe(df.head())

    # Attempt to load models & metadata from working folder
    # Priority: local 'models' folder, otherwise a 'models' folder in current directory
    # The notebook that creates these files should place them in ./models (or you can edit path).
    model_files = {}
    for p in ["./models", "./"]:
        try:
            cols_info = joblib.load(Path(p) / 'cols_info.joblib')
            MODELS_DIR = Path(p)
            break
        except Exception as e:
            cols_info = None
    if cols_info is None:
        st.error("cols_info.joblib not found. Please ensure the models folder contains cols_info.joblib (feature list & target col).")
    else:
        feature_columns = cols_info['feature_columns']
        st.write("Expected feature columns (from training):", feature_columns)

        # Keep only expected features if present
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            st.warning(f"Missing expected features: {missing}. The app will try to use the columns available.")
        X = df.reindex(columns=feature_columns).copy()
        # If some columns are missing, they'll be NaN; attempt to fill
        X = X.fillna(X.median())

        # Load scaler
        scaler = joblib.load(Path(MODELS_DIR) / 'scaler.joblib')
        X_scaled = scaler.transform(X.values)

        # Load all models (files ending with _model.pkl)
        import glob
        model_paths = glob.glob(str(Path(MODELS_DIR) / "*_model.pkl"))
        if not model_paths:
            st.error("No model files (*.pkl) found in models folder.")
        else:
            st.success(f"Found {len(model_paths)} model files.")
            predictions = {}
            for mp in model_paths:
                name = Path(mp).stem.replace("_model", "")
                try:
                    with open(mp, 'rb') as f:
                        model = pickle.load(f)
                    preds = model.predict(X_scaled)
                    # If label encoder exists, try to inverse transform
                    try:
                        le = joblib.load(Path(MODELS_DIR) / 'label_encoder.joblib')
                        preds = le.inverse_transform(preds)
                    except:
                        pass
                    predictions[name] = preds
                except Exception as e:
                    st.warning(f"Could not load/predict with {mp}: {e}")

            # Compose results DataFrame
            results_df = df.copy().reset_index(drop=True)
            for name, preds in predictions.items():
                results_df[f'pred_{name}'] = preds

            st.write("Predictions (first 10 rows):")
            st.dataframe(results_df.head(10))

            # Allow download of results CSV
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", data=csv, file_name="predictions_with_models.csv", mime="text/csv")

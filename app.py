# =============================================================================
# --- app.py: Streamlit Web Application ---
# =============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os

# --- Configuration ---
MODEL_PATH = 'ecg_mlp_model.h5'
SCALER_PATH = 'scaler.pkl'
SEGMENT_LENGTH = 187 # Match the segment size of the training data
SAMPLE_RATE = 125 # The Kaggle dataset uses 125 Hz

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    """Loads the trained model and scaler object."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error: Model ('{MODEL_PATH}') or Scaler ('{SCALER_PATH}') not found.")
        st.warning("Please run ecg_arrhythmia_detector.py first.")
        return None, None

# --- Feature Extraction (Must match the training script!) ---
def extract_features(beat_segment):
    """Calculates simple statistical features (Mean, Std Dev, Min)."""
    try:
        mean = np.mean(beat_segment)
        std = np.std(beat_segment)
        min_val = np.min(beat_segment)
        
        return np.array([mean, std, min_val]).reshape(1, -1)
    except Exception:
        return None

# --- Main Streamlit App ---
st.set_page_config(page_title="Chaotic Feature ECG Arrhythmia Detector", layout="wide")

st.title("Chaotic Feature ECG Arrhythmia Detector 💖")
st.markdown("A simple Multi-Layer Perceptron (MLP) trained on features like Mean, Std Dev, and Minimum value.")

model, scaler = load_assets()

if model and scaler:
    st.sidebar.header("Upload ECG Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file containing a single ECG signal column.", type="csv")

    st.sidebar.markdown("""
        **Data Requirements:**
        - Single column time series.
        - Must contain at least 2 seconds of signal (250 samples).
        """)

    if uploaded_file is not None:
        try:
            # Read the CSV (assuming single column)
            signal_df = pd.read_csv(uploaded_file, header=None)
            full_signal = signal_df.iloc[:, 0].values

            st.header("1. Uploaded Signal Visualization")
            st.line_chart(full_signal, use_container_width=True)
            st.write(f"Signal Length: {len(full_signal)} samples (Approx. {len(full_signal)/SAMPLE_RATE:.2f} seconds)")

            if len(full_signal) < SEGMENT_LENGTH:
                st.warning(f"Signal is too short. Need at least {SEGMENT_LENGTH} samples.")
            else:
                # --- Segmentation (Take a centered segment) ---
                center = len(full_signal) // 2
                start = center - SEGMENT_LENGTH // 2
                end = center + SEGMENT_LENGTH // 2
                
                # Use a segment of the full signal for prediction
                segment = full_signal[start:end]

                # --- Prediction Pipeline ---
                
                # 1. Extract Features
                features = extract_features(segment)
                if features is None:
                    st.error("Feature extraction failed for the signal segment.")
                else:
                    # 2. Scale Features
                    features_scaled = scaler.transform(features)
                    
                    # 3. Predict
                    prediction_proba = model.predict(features_scaled, verbose=0)[0][0]
                    
                    # 4. Display Results
                    st.header("2. Prediction Result")
                    
                    if prediction_proba > 0.5:
                        result_text = "ARRHYTHMIA DETECTED (Abnormal)"
                        st.error(f"Prediction: {result_text}")
                    else:
                        result_text = "NORMAL RHYTHM (Healthy)"
                        st.success(f"Prediction: {result_text}")
                        
                    st.metric(label="Arrhythmia Probability", value=f"{prediction_proba * 100:.2f}%")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
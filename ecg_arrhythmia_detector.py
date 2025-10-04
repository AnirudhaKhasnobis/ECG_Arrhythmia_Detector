# =============================================================================
# --- ecg_arrhythmia_detector.py: MODEL TRAINING SCRIPT (KAGGLE CSV FINAL) ---
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
import pickle
import warnings
import os

# Suppress minor warnings, including future TensorFlow and nolds warnings
warnings.filterwarnings('ignore') 

# --- Configuration ---
INPUT_FEATURES = 187 # Number of samples in the pre-segmented heartbeat
FEATURE_COUNT = 3 # We are using 3 statistical features: Mean, Std, Min

# =============================================================================
# --- Core Processing Functions ---
# =============================================================================

def load_and_prepare_data(train_file, test_file):
    """Loads the train/test CSVs and converts the 5-class problem to Binary (Normal vs. Arrhythmia)."""
    
    # 1. Load DataFrames
    try:
        train_df = pd.read_csv(train_file, header=None)
        test_df = pd.read_csv(test_file, header=None)
        
        print("  [Setup] CSV Data Loaded successfully.")
        
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Could not find '{train_file}' or '{test_file}'.")
        print("Please ensure you have downloaded and copied the CSV files from Kaggle.")
        return None, None, None, None, False
        
    # 2. Separate features (X) and labels (y)
    X_train = train_df.iloc[:, :INPUT_FEATURES].values
    y_train = train_df.iloc[:, INPUT_FEATURES].values
    X_test = test_df.iloc[:, :INPUT_FEATURES].values
    y_test = test_df.iloc[:, INPUT_FEATURES].values
    
    # 3. Convert 5-class problem (0, 1, 2, 3, 4) into a Binary problem (0=Normal, 1=Arrhythmia)
    # 0 is Normal. Any other label (1, 2, 3, 4) is considered Arrhythmia (1).
    y_train_binary = np.where(y_train == 0, 0, 1)
    y_test_binary = np.where(y_test == 0, 0, 1)

    print(f"  [Setup] Training Set Size: {len(X_train)} heartbeats.")
    
    return X_train, X_test, y_train_binary, y_test_binary, True


def extract_features(beat_segment):
    """Calculates simple statistical features (Mean, Std Dev, Min)."""
    try:
        mean = np.mean(beat_segment)
        std = np.std(beat_segment)
        min_val = np.min(beat_segment)
        
        # We return 3 features to match the network input size (3)
        return [mean, std, min_val]
    except Exception:
        return None

def build_mlp_model(input_shape):
    """Defines the Keras Multi-Layer Perceptron (Neural Network)."""
    model = keras.Sequential([
        # Input shape is 3, corresponding to [Mean, Std, Min]
        keras.layers.Dense(16, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =============================================================================
# --- Main Execution Block for Training ---
# =============================================================================
if __name__ == "__main__":
    
    TRAIN_FILE = 'mitbih_train.csv'
    TEST_FILE = 'mitbih_test.csv'

    print("\n--- 1. Starting Data Loading from CSV ---")
    X_train_raw, X_test_raw, y_train, y_test, success = load_and_prepare_data(TRAIN_FILE, TEST_FILE)
    
    if not success:
        exit()
        
    print("\n--- 2. Extracting Features (This may take a few minutes) ---")
    
    # --- Feature Extraction ---
    X_train_features = []
    y_train_filtered = []
    
    for beat_signal, label in zip(X_train_raw, y_train):
        features = extract_features(beat_signal)
        if features is not None:
            X_train_features.append(features)
            y_train_filtered.append(label)

    X_test_features = []
    y_test_filtered = []
    
    for beat_signal, label in zip(X_test_raw, y_test):
        features = extract_features(beat_signal)
        if features is not None:
            X_test_features.append(features)
            y_test_filtered.append(label)

    X_train = np.array(X_train_features)
    X_test = np.array(X_test_features)
    y_train = np.array(y_train_filtered)
    y_test = np.array(y_test_filtered)
    
    if len(X_train) < 100:
        print("\nERROR: Not enough features extracted. Check the feature extraction function.")
        exit()

    print(f"--- 3. Feature Extraction Complete: {len(X_train)} training heartbeats used ---")
    
    # --- Step 4: Model Training and Saving ---
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("--- 4. Training MLP Model (50 Epochs) ---")
    model = build_mlp_model(input_shape=FEATURE_COUNT) 
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=256, validation_split=0.1, verbose=1)
    
    # Save Model and Scaler
    print("\n--- 5. Saving Assets ---")
    model.save('ecg_mlp_model.h5')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("  'ecg_mlp_model.h5' and 'scaler.pkl' saved successfully.")
    
    # Final Evaluation Summary
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"  Test Accuracy:    {accuracy * 100:.2f}%")
    print("\n--- Training Finished ---")
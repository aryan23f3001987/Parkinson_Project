import pickle
import pandas as pd
from feature_extractors.classification_features import extract_classification_features
from preprocessing.scaler import scale_features

def predict_parkinson(audio_path="audio_samples/audio.wav"):
    features_df = extract_classification_features(audio_path)

    scaled_df = scale_features(features_df)

    with open("models/classification.pkl", "rb") as f:
        model = pickle.load(f)
        
    y_pred = model.predict(scaled_df)[0]
    return "Parkinson Detected" if y_pred == 1 else "Healthy"


if __name__ == "__main__":
    result = predict_parkinson("audio_samples/audio.wav")
    print("🧠 Prediction:", result)

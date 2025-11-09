import pandas as pd
import numpy as np

def align_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Ensures the extracted DataFrame has the same columns
    (in the same order) as the training dataset used for model training.
    Missing columns are filled with 0.
    """
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_list]


def log_message(message: str):
    """
    Pretty print with a uniform style for console logs.
    """
    print(f"🔹 {message}")


def check_audio_quality(duration, sample_rate):
    """
    Optional: Check if the audio is long enough and valid for feature extraction.
    """
    if duration < 1.0:
        return "⚠️ Audio too short — try recording longer."
    elif sample_rate < 16000:
        return "⚠️ Low sampling rate — please record with 16kHz or higher."
    return "✅ Audio quality OK."

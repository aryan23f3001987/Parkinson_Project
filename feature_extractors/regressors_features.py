import parselmouth
import numpy as np
import pandas as pd

def extract_regression_features(audio_path: str, age=65, sex=0, test_time=100.0) -> pd.DataFrame:
    """
    Extracts all features required by the regression models
    (motor_UPDRS, total_UPDRS) from an audio file.
    """
    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)

    # Jitter & Shimmer features
    jitter_percent = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 75, 500, 1.3)
    jitter_abs = parselmouth.praat.call(pulses, "Get jitter (local, absolute)", 0, 0, 75, 500, 1.3)
    rap = parselmouth.praat.call(pulses, "Get jitter (rap)", 0, 0, 75, 500, 1.3)
    ppq = parselmouth.praat.call(pulses, "Get jitter (ppq5)", 0, 0, 75, 500, 1.3)
    ddp = 3 * rap

    shimmer = parselmouth.praat.call([sound, pulses], "Get shimmer (local)", 0, 0, 75, 500, 1.3, 1.6)
    shimmer_db = parselmouth.praat.call([sound, pulses], "Get shimmer (local_dB)", 0, 0, 75, 500, 1.3, 1.6)
    apq3 = parselmouth.praat.call([sound, pulses], "Get shimmer (apq3)", 0, 0, 75, 500, 1.3, 1.6)
    apq5 = parselmouth.praat.call([sound, pulses], "Get shimmer (apq5)", 0, 0, 75, 500, 1.3, 1.6)
    apq11 = parselmouth.praat.call([sound, pulses], "Get shimmer (apq11)", 0, 0, 75, 500, 1.3, 1.6)
    dda = 3 * apq3

    # Harmonics
    hnr_obj = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    HNR = parselmouth.praat.call(hnr_obj, "Get mean", 0, 0)
    NHR = 1 / (1 + 10 ** (HNR / 10))

    # Nonlinear dynamic measures (RPDE, DFA, PPE)
    # rpde = parselmouth.praat.call(sound, "Get entropy (Spectral)", 0, 0)
    # dfa = parselmouth.praat.call(sound, "Get standard deviation", 0, 0)
    # ppe = abs(np.log10(np.mean([jitter_percent + shimmer])))

    rpde = 0.0   # Placeholder – true extraction requires nonlinear analysis
    dfa = 0.0
    ppe = abs(np.log10(np.mean([jitter_percent + shimmer])))

    features = {
        "age": age,
        "sex": sex,
        "test_time": test_time,
        "Jitter(%)": jitter_percent,
        "Jitter(Abs)": jitter_abs,
        "Jitter:RAP": rap,
        "Jitter:PPQ5": ppq,
        "Jitter:DDP": ddp,
        "Shimmer": shimmer,
        "Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": apq5,
        "Shimmer:APQ11": apq11,
        "Shimmer:DDA": dda,
        "NHR": NHR,
        "HNR": HNR,
        "RPDE": rpde,
        "DFA": dfa,
        "PPE": ppe
    }

    df = pd.DataFrame([features])
    return df


if __name__ == "__main__":
    df = extract_regression_features("audio_samples/audio.wav", age=70, sex=0, test_time=50.0)

    # Save the extracted features for pipeline use
    save_path = "data_storage/regression_features.csv"
    df.to_csv(save_path, index=False)
    print(f"✅ Regression features saved at: {save_path}")

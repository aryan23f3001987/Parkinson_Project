import parselmouth
import numpy as np
import pandas as pd

def extract_classification_features(audio_path: str) -> pd.DataFrame:
    """
    Extracts all features required by the classification model
    from an input audio file using Parselmouth (Praat).
    """
    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)

    Fo = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
    Fhi = parselmouth.praat.call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    Flo = parselmouth.praat.call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

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

    hnr = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    HNR = parselmouth.praat.call(hnr, "Get mean", 0, 0)

    NHR = 1 / (1 + 10 ** (HNR / 10))

    features = {
        "Fo(Hz)": Fo,
        "Fhi(Hz)": Fhi,
        "Flo(Hz)": Flo,
        "Jitter(%)": jitter_percent,
        "Jitter(Abs)": jitter_abs,
        "Jitter:RAP": rap,
        "Jitter:PPQ": ppq,
        "Jitter:DDP": ddp,
        "Shimmer": shimmer,
        "Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": apq5,
        "Shimmer:APQ11": apq11,
        "Shimmer:DDA": dda,
        "NHR": NHR,
        "HNR": HNR,
    }

    df = pd.DataFrame([features])
    return df


if __name__ == "__main__":
    df = extract_classification_features("audio_samples/audio.wav")

    save_path = "data_storage/classification_features.csv"
    df.to_csv(save_path, index=False)
    print(f"✅ Classification features saved at: {save_path}")

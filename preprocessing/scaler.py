import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(df: pd.DataFrame, scaler_path: str = "models/scaler.pkl") -> pd.DataFrame:
    """
    Scales the extracted feature DataFrame using StandardScaler.
    Saves scaler if not present, else loads it.
    """
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = StandardScaler()
        scaler.fit(df)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print("✅ New scaler fitted and saved.")

    scaled_data = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    return scaled_df
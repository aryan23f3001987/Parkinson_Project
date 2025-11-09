import os
import pickle
import joblib

def load_model(model_name: str):
    """
    Loads a .pkl model from the models directory.
    Supports both pickle and joblib formats.
    """
    model_path = os.path.join("models", model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Loaded model (pickle): {model_name}")
    except Exception:
        model = joblib.load(model_path)
        print(f"✅ Loaded model (joblib): {model_name}")

    return model

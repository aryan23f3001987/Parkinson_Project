import pandas as pd
from feature_extractors.classification_features import extract_classification_features
from feature_extractors.regressors_features import extract_regression_features
from preprocessing.scaler import scale_features
from utils.load_models import load_model
import constants

# Audio input path
AUDIO_PATH = "audio_samples/audio.wav"

def run_pipeline(audio_path=AUDIO_PATH, age=70, sex=0, test_time=50.0):
    print("\n🎙️ Starting Parkinson Prediction Pipeline...\n")

    # 1️⃣ Extract Classification Features
    classification_df = extract_classification_features(audio_path)
    classification_df.to_csv("data_storage/classification_features.csv", index=False)
    print("✅ Classification features extracted and saved.")

    # 2️⃣ Scale classification features
    classification_scaled = scale_features(classification_df, "models\scaler_classification.pkl")

    # 3️⃣ Load classification model and predict
    clf_model = load_model("classification.pkl")
    clf_pred = clf_model.predict(classification_scaled)[0]
    constants.PARKINSON_STATUS = "Parkinson Detected" if clf_pred == 1 else "Healthy"
    print(f"🧠 Classification Result: {constants.PARKINSON_STATUS}")

    # 4️⃣ If Parkinson detected, run regression models
    if clf_pred == 1:
        regression_df = extract_regression_features(audio_path, age, sex, test_time)
        regression_df.to_csv("data_storage/regression_features.csv", index=False)
        print("✅ Regression features extracted and saved.")

        regression_scaled = scale_features(regression_df, "models\scaler_regression.pkl")

        # Load regression models
        motor_model = load_model("motor_updrs_model.pkl")
        total_model = load_model("total_updrs_model.pkl")

        # Predict UPDRS scores
        constants.MOTOR_UPDRS_SCORE = round(motor_model.predict(regression_scaled)[0], 2)
        constants.TOTAL_UPDRS_SCORE = round(total_model.predict(regression_scaled)[0], 2)

        print(f"🎯 Motor UPDRS Score: {constants.MOTOR_UPDRS_SCORE}")
        print(f"🎯 Total UPDRS Score: {constants.TOTAL_UPDRS_SCORE}")
    else:
        constants.MOTOR_UPDRS_SCORE = 0.0
        constants.TOTAL_UPDRS_SCORE = 0.0
        print("🩵 Healthy subject, skipping UPDRS prediction.")

    print("\n🚀 Pipeline Complete!")
    print("----------------------------------")


if __name__ == "__main__":
    run_pipeline()

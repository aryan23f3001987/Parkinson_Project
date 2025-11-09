import pickle
from feature_extractors.regressors_features import extract_regression_features
from preprocessing.scaler import scale_features

def predict_updrs(audio_path="audio_samples/audio.wav", age=65, sex=0, test_time=100.0):
    df = extract_regression_features(audio_path, age, sex, test_time)

    scaled_df = scale_features(df)

    with open("models/motor_updrs_model.pkl", "rb") as f:
        motor_model = pickle.load(f)

    with open("models/total_updrs_model.pkl", "rb") as f:
        total_model = pickle.load(f)

    motor_pred = motor_model.predict(scaled_df)[0]
    total_pred = total_model.predict(scaled_df)[0]

    return {
        "Motor UPDRS": round(motor_pred, 3),
        "Total UPDRS": round(total_pred, 3)
    }


if __name__ == "__main__":
    scores = predict_updrs("audio_samples/audio.wav", age=70, sex=0, test_time=90.0)
    print("🎯 Predicted UPDRS Scores:", scores)

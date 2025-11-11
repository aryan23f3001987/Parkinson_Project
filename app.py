# from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# from feature_extractors.classification_features import extract_classification_features
# from feature_extractors.regressors_features import extract_regression_features
# from preprocessing.scaler import scale_features
# from utils.load_models import load_model
# import constants
# import os

# app = Flask(__name__)
# CORS(app)

# # Audio input path
# AUDIO_PATH = "audio_samples/audio.wav"

# def run_pipeline(audio_path=AUDIO_PATH, age=70, sex=0, test_time=50.0):
#     print("\n🎙️ Starting Parkinson Prediction Pipeline...\n")

#     # 1️⃣ Extract Classification Features
#     classification_df = extract_classification_features(audio_path)
#     classification_df.to_csv("data_storage/classification_features.csv", index=False)
#     print("✅ Classification features extracted and saved.")

#     # 2️⃣ Scale classification features
#     classification_scaled = scale_features(classification_df, "models\\scaler_classification.pkl")

#     # 3️⃣ Load classification model and predict
#     clf_model = load_model("classification.pkl")

#     # Check if model supports probability
#     if hasattr(clf_model, "predict_proba"):
#         proba = clf_model.predict_proba(classification_scaled)[0][1]  # probability of Parkinson
#         print(f"🧩 Parkinson Probability: {proba:.3f}")
#         clf_pred = 1 if proba > 0.6 else 0  # threshold = 0.6 (tweakable)
#     else:
#         print("⚠️ Model does not support probability. Using direct prediction.")
#         clf_pred = clf_model.predict(classification_scaled)[0]

#     constants.PARKINSON_STATUS = "Parkinson Detected" if clf_pred == 1 else "Healthy"
#     print(f"🧠 Classification Result: {constants.PARKINSON_STATUS}")


#     if clf_pred == 1:
#         regression_df = extract_regression_features(audio_path, age, sex, test_time)
#         print("✅ Regression features extracted and saved.")

#         regression_scaled = scale_features(regression_df, "models\\scaler_regression.pkl")

#         motor_model = load_model("motor_updrs_model.pkl")
#         total_model = load_model("total_updrs_model.pkl")

#         constants.MOTOR_UPDRS_SCORE = round(motor_model.predict(regression_scaled)[0], 2)
#         constants.TOTAL_UPDRS_SCORE = round(total_model.predict(regression_scaled)[0], 2)

#         print(f"🎯 Motor UPDRS Score: {constants.MOTOR_UPDRS_SCORE}")
#         print(f"🎯 Total UPDRS Score: {constants.TOTAL_UPDRS_SCORE}")
#     else:
#         constants.MOTOR_UPDRS_SCORE = 0.0
#         constants.TOTAL_UPDRS_SCORE = 0.0
#         print("🩵 Healthy subject, skipping UPDRS prediction.")

#     print("\n🚀 Pipeline Complete!")
#     print("----------------------------------")


# @app.route('/')
# def index():
#     """Serve the main HTML page"""
#     return render_template('index.html')


# @app.route('/analyze', methods=['POST'])
# def analyze():
#     """API endpoint to analyze audio"""
#     try:
#         audio_file = request.files.get('audio')
#         if not audio_file:
#             return jsonify({'error': 'No audio file provided'}), 400

#         # ✅ Get user-provided data
#         age = int(request.form.get('age', 70))  # default 70 if not provided
#         sex_str = request.form.get('sex', 'male').lower()
#         test_time = float(request.form.get('test_time', 50.0))

#         # Convert sex to numeric (match model training)
#         sex = 1 if sex_str in ['male', 'm', '1'] else 0

#         # Save uploaded audio file
#         audio_save_path = os.path.join('audio_samples', 'audio.wav')
#         if os.path.exists(audio_save_path):
#             os.remove(audio_save_path)
#         audio_file.save(audio_save_path)
#         print(f"💾 New audio saved: {audio_save_path}")

#         # Run the full pipeline
#         run_pipeline(audio_save_path, age, sex, test_time)

#         # Return JSON results
#         return jsonify({
#             'status': constants.PARKINSON_STATUS,
#             'motor_updrs': constants.MOTOR_UPDRS_SCORE,
#             'total_updrs': constants.TOTAL_UPDRS_SCORE
#         })

#     except Exception as e:
#         print(f"❌ Error during analysis: {str(e)}")
#         return jsonify({'error': str(e)}), 500



# if __name__ == "__main__":
#     # Ensure required directories exist
#     os.makedirs("data_storage", exist_ok=True)
#     os.makedirs("audio_samples", exist_ok=True)
#     os.makedirs("templates", exist_ok=True)
    
#     print("🚀 Starting Flask Server...")
#     print(f"📁 Make sure index.html is in: {os.path.abspath('templates')}")
#     print("📱 Open http://localhost:5000 in your browser")
#     print("----------------------------------")
    
#     app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from feature_extractors.classification_features import extract_classification_features
from feature_extractors.regressors_features import extract_regression_features
from preprocessing.scaler import scale_features
from utils.load_models import load_model
import constants
import os
import time
import librosa
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

AUDIO_PATH = "audio_samples/audio.wav"


def convert_to_wav(input_path, output_path):
    """Convert uploaded audio (mp3, m4a, etc.) to WAV format."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        print(f"🎧 Converted to WAV: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Audio conversion failed: {e}")


def calculate_test_time(audio_path):
    """Calculate duration of audio in seconds."""
    try:
        y, sr = librosa.load(audio_path)
        duration = round(len(y) / sr, 2)
        print(f"🕒 Extracted test_time: {duration} seconds")
        return duration
    except Exception as e:
        print(f"⚠️ Failed to calculate duration: {e}")
        return 10.0  # fallback


def clean_audio_folder():  # 🧹 NEW FUNCTION
    """Delete all previous audio files each time the page is refreshed."""
    folder = "audio_samples"
    deleted = 0
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted += 1
    print(f"🧹 Cleaned {deleted} old audio files.")


def run_pipeline(audio_path=AUDIO_PATH, age=70, sex=0, test_time=50.0):
    print("\n🎙️ Starting Parkinson Prediction Pipeline...\n")

    # 1️⃣ Classification
    classification_df = extract_classification_features(audio_path)
    classification_df.to_csv("data_storage/classification_features.csv", index=False)
    print("✅ Classification features extracted.")

    classification_scaled = scale_features(classification_df, "models/scaler_classification.pkl")
    clf_model = load_model("classification.pkl")

    if hasattr(clf_model, "predict_proba"):
        proba = clf_model.predict_proba(classification_scaled)[0][1]
        print(f"🧩 Parkinson Probability: {proba:.3f}")
    else:
        proba = float(clf_model.predict(classification_scaled)[0])
        print(f"⚠️ Model does not support probability — using direct prediction: {proba}")

    # 2️⃣ Regression features (common)
    regression_df = extract_regression_features(audio_path, age, sex, test_time)
    print("✅ Regression features extracted.")

    # Separate dataframes for with-age and without-age models
    regression_with_age = regression_df.copy()
    regression_without_age = regression_df.drop(columns=["age"])

    # 3️⃣ Scale using respective scalers
    scaled_with_age = scale_features(regression_with_age, "models/scaler_regression_age.pkl")
    scaled_without_age = scale_features(regression_without_age, "models/scaler_regression_without_age.pkl")

    # 4️⃣ Load both model sets
    motor_with_age = load_model("motor_updrs_model_age.pkl")
    motor_without_age = load_model("motor_updrs_model_without_age.pkl")
    total_with_age = load_model("total_updrs_model_age.pkl")
    total_without_age = load_model("total_updrs_model_without_age.pkl")

    # 5️⃣ Predictions
    motor_pred_age = motor_with_age.predict(scaled_with_age)[0]
    motor_pred_wo_age = motor_without_age.predict(scaled_without_age)[0]
    total_pred_age = total_with_age.predict(scaled_with_age)[0]
    total_pred_wo_age = total_without_age.predict(scaled_without_age)[0]

    # Weighted ensemble (less weight for with-age models)
    constants.MOTOR_UPDRS_SCORE = round((0.35 * motor_pred_age + 0.65 * motor_pred_wo_age), 2)
    constants.TOTAL_UPDRS_SCORE = round((0.35 * total_pred_age + 0.65 * total_pred_wo_age), 2)

    print(f"🎯 Motor UPDRS (ensemble): {constants.MOTOR_UPDRS_SCORE}")
    print(f"🎯 Total UPDRS (ensemble): {constants.TOTAL_UPDRS_SCORE}")

    # 6️⃣ Final decision based on hybrid logic
    motor = constants.MOTOR_UPDRS_SCORE
    total = constants.TOTAL_UPDRS_SCORE

    if proba < 0.4 and motor < 8 and total < 10:
        constants.PARKINSON_STATUS = "Healthy"
    elif 0.4 <= proba < 0.65 or (8 <= motor < 15 or 10 <= total < 20):
        constants.PARKINSON_STATUS = "Minor Parkinson"
    elif 0.65 <= proba < 0.85 or (15 <= motor < 25 or 20 <= total < 35):
        constants.PARKINSON_STATUS = "Moderate Parkinson"
    else:
        constants.PARKINSON_STATUS = "Severe Parkinson"

    print(f"🧠 Final Assessment: {constants.PARKINSON_STATUS}")
    print("\n🚀 Pipeline Complete!\n----------------------------------")

    return proba


@app.route('/')
def index():
    clean_audio_folder()  # 🧹 Automatically clean old files on refresh or first visit
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400

        # ✅ Get user-provided data
        age = int(request.form.get('age', 70))
        sex_str = request.form.get('sex', 'male').lower()
        user_test_time = request.form.get('test_time', None)
        sex = 1 if sex_str in ['male', 'm', '1'] else 0

        # Save uploaded/recorded audio
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        raw_audio_path = os.path.join('audio_samples', f"audio_raw_{int(time.time())}{file_ext}")
        wav_audio_path = os.path.join('audio_samples', f"audio_{int(time.time())}.wav")
        audio_file.save(raw_audio_path)
        print(f"💾 Audio saved: {raw_audio_path}")

        # Convert to WAV if needed
        if file_ext != ".wav":
            convert_to_wav(raw_audio_path, wav_audio_path)
        else:
            wav_audio_path = raw_audio_path

        # Determine test_time
        if user_test_time:
            test_time = float(user_test_time)
            print(f"🧮 Using user-provided test_time: {test_time}")
        else:
            test_time = calculate_test_time(wav_audio_path)

        # Run prediction pipeline
        proba = run_pipeline(wav_audio_path, age, sex, test_time)

        response = jsonify({
            'status': constants.PARKINSON_STATUS,
            'probability': round(proba, 3),
            'motor_updrs': constants.MOTOR_UPDRS_SCORE,
            'total_updrs': constants.TOTAL_UPDRS_SCORE,
            'test_time': test_time
        })
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    os.makedirs("data_storage", exist_ok=True)
    os.makedirs("audio_samples", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    print("🚀 Starting Flask Server...")
    print(f"📁 Make sure index.html is in: {os.path.abspath('templates')}")
    print("📱 Open http://localhost:5000 in your browser")
    print("----------------------------------")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

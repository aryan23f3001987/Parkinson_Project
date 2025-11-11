<div align="center">
# 🎙️ Parkinson's Disease Detection System

An AI-powered web application that analyzes voice recordings to detect potential signs of Parkinson's Disease using machine learning models. The system provides real-time audio visualization and detailed analysis including UPDRS scores and probability assessments.

![Parkinson's Detection System](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange)
</div>

## 📋 Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Model Information](#model-information)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

## ✨ Features

- **Dual Input Modes**
  - 🎤 **Live Recording**: Record audio directly from your microphone
  - 📁 **File Upload**: Upload pre-recorded audio files (WAV, MP3, M4A)

- **Real-Time Audio Visualization**
  - Beautiful animated waveforms with particle effects
  - Live frequency bars during recording
  - Gradient color schemes with glowing peak indicators

- **Comprehensive Analysis**
  - Parkinson's Disease probability assessment
  - Motor UPDRS score prediction
  - Total UPDRS score prediction
  - Severity classification (Healthy, Minor, Moderate, Severe)

- **User-Friendly Interface**
  - Modern dark theme with glassmorphism effects
  - Responsive design for all devices
  - Intuitive drag-and-drop file upload
  - Real-time feedback and status updates

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep learning models
- **NumPy** - Numerical computations
- **Librosa** - Audio feature extraction
- **Pydub** - Audio file processing
- **SciPy** - Scientific computing

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with modern animations
- **Vanilla JavaScript** - Interactive functionality
- **Canvas API** - Audio visualization
- **Web Audio API** - Audio processing

## 📦 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/parkinsons-detection.git
cd parkinsons-detection
```

### Step 2: Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install System Dependencies

**For Windows:**
- Download and install [FFmpeg](https://ffmpeg.org/download.html)
- Add FFmpeg to system PATH

**For macOS:**
```bash
brew install ffmpeg
```

**For Linux:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### Step 5: Place Model Files
Ensure your trained model files are in the project directory:
- `parkinsons_classification_model.h5` - Classification model
- `parkinsons_motor_updrs_model.h5` - Motor UPDRS prediction model
- `parkinsons_total_updrs_model.h5` - Total UPDRS prediction model

### Step 6: Run the Application
```bash
python app.py
```

The application will start on `http://127.0.0.1:5000`

## 🚀 Usage

### Recording Audio
1. Open the application in your web browser
2. Enter your age and select gender
3. Set recording duration (5-60 seconds)
4. Click "Start Recording" and speak clearly
5. Wait for analysis results

### Uploading Audio
1. Switch to "Upload Audio" mode
2. Enter your age and select gender
3. Click the upload zone or drag & drop an audio file
4. Click "Analyze Audio"
5. View results

### Best Practices for Recording
- **Environment**: Record in a quiet environment
- **Distance**: Keep microphone 6-12 inches from mouth
- **Speech**: Speak normally and clearly
- **Duration**: Minimum 5 seconds recommended
- **Content**: Sustained vowel sounds or reading passages work best

## 🔬 How It Works

### Audio Processing Pipeline

1. **Audio Input**
   - Capture from microphone or upload file
   - Convert to WAV format (16-bit, mono)

2. **Feature Extraction**
   - Load audio with Librosa
   - Extract acoustic features:
     - **MFCC** (Mel-Frequency Cepstral Coefficients)
     - **Chroma** features
     - **Mel Spectrogram**
     - **Spectral Contrast**
     - **Tonnetz** (Tonal Centroid Features)
   - Statistical aggregation (mean and standard deviation)

3. **Preprocessing**
   - Normalize features
   - Handle missing values
   - Reshape for model input

4. **Model Prediction**
   - Classification model predicts Parkinson's probability
   - Regression models predict UPDRS scores
   - Combine predictions for final assessment

5. **Result Presentation**
   - Display probability percentage
   - Show UPDRS scores
   - Classify severity level

### Feature Engineering

The system extracts **40 key audio features**:
- 20 MFCC coefficients (mean + std)
- 12 Chroma features (mean + std)
- 128 Mel spectrogram bands (mean + std)
- 7 Spectral contrast bands (mean + std)
- 6 Tonnetz features (mean + std)

## 📁 Project Structure

```
parkinsons-detection/
│
├── app.py                                      # Flask backend server
├── index.html                                  # Frontend UI
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
│
├── models/                                     # Trained ML models
│   ├── parkinsons_classification_model.h5
│   ├── parkinsons_motor_updrs_model.h5
│   └── parkinsons_total_updrs_model.h5
│
├── uploads/                                    # Temporary audio storage
│   └── (auto-generated during runtime)
│
└── static/                                     # Static assets (optional)
    └── (icons, images, etc.)
```

## 🔌 API Endpoints

### POST `/analyze`
Analyzes uploaded audio file for Parkinson's detection.

**Request:**
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Body**:
  - `audio` (file): Audio file (WAV, MP3, M4A, etc.)
  - `age` (string): Patient age
  - `sex` (string): "male" or "female"
  - `test_time` (string, optional): Recording duration

**Response:**
```json
{
  "status": "Healthy | Parkinson's Detected - Minor | Moderate | Severe",
  "probability": 0.85,
  "motor_updrs": 15.3,
  "total_updrs": 28.7
}
```

**Error Response:**
```json
{
  "error": "Error message description"
}
```

## 🤖 Model Information

### Classification Model
- **Architecture**: Deep Neural Network
- **Input**: 193 audio features
- **Output**: Binary classification (0 = Healthy, 1 = Parkinson's)
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### UPDRS Prediction Models
- **Motor UPDRS**: Predicts movement symptom severity (0-108 scale)
- **Total UPDRS**: Predicts overall symptom severity (0-176 scale)
- **Architecture**: Regression Neural Networks

### UPDRS Score Interpretation
- **0-20**: Minimal symptoms
- **21-40**: Mild symptoms
- **41-60**: Moderate symptoms
- **61+**: Severe symptoms

## 🐛 Troubleshooting

### Common Issues

**Issue: "Microphone access denied"**
- **Solution**: Grant microphone permissions in browser settings
- **Chrome**: Settings → Privacy and Security → Site Settings → Microphone
- **Firefox**: Preferences → Privacy & Security → Permissions → Microphone

**Issue: "Module not found" errors**
- **Solution**: Ensure virtual environment is activated and all dependencies installed
```bash
pip install -r requirements.txt
```

**Issue: "FFmpeg not found"**
- **Solution**: Install FFmpeg and add to system PATH
- Verify installation: `ffmpeg -version`

**Issue: Audio upload fails**
- **Solution**: Check file format (must be audio file)
- Maximum file size: 50MB
- Supported formats: WAV, MP3, M4A, OGG, FLAC

**Issue: Inaccurate predictions**
- **Solution**: 
  - Ensure correct age input
  - Record in quiet environment
  - Speak clearly for at least 5 seconds
  - Use sustained vowel sounds for best results

**Issue: Models not loading**
- **Solution**: Verify model files exist in correct location
- Check file permissions
- Ensure model files are not corrupted

## ⚠️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER:**

This application is an AI-powered research tool and is **NOT** a substitute for professional medical advice, diagnosis, or treatment. 

- ❌ Do NOT use this tool as the sole basis for medical decisions
- ❌ Do NOT replace consultation with qualified healthcare providers
- ❌ Do NOT delay seeking professional medical advice based on results

**This tool should be used for:**
- ✅ Educational purposes
- ✅ Research and development
- ✅ Preliminary screening (followed by professional evaluation)
- ✅ Supplementary information for healthcare discussions

**Always consult with a qualified neurologist or healthcare provider for:**
- Proper diagnosis and evaluation
- Treatment planning
- Medical advice and guidance
- Interpretation of symptoms

The accuracy of predictions depends on various factors including recording quality, user age accuracy, and model limitations. False positives and false negatives are possible.

## 👥 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions, issues, or suggestions:
- **Email**: your.email@example.com
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/parkinsons-detection/issues)

## 🙏 Acknowledgments

- Audio feature extraction powered by [Librosa](https://librosa.org/)
- Deep learning models built with [TensorFlow](https://www.tensorflow.org/)
- UI/UX inspired by modern design principles
- Medical knowledge based on UPDRS (Unified Parkinson's Disease Rating Scale)

## 📊 Development Roadmap

- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Enhanced model accuracy
- [ ] Longitudinal tracking features
- [ ] Integration with electronic health records
- [ ] Advanced analytics dashboard
- [ ] Voice exercise recommendations

---

**Made with ❤️ for Parkinson's Disease awareness and early detection**

**Version**: 1.0.0  
**Last Updated**: November 2025
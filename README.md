<div align="center">

```text
  _____ __  __  ___ _____ ___ ___  _   _    _   ___ 
 | ____|  \/  |/ _ \_   _|_ _| _ \| \ | |  / \ |_ _|
 |  _| | |\/| | | | || |  | || |_) |  \| | / _ \ | | 
 | |___| |  | | |_| || |  | ||  _ <| |\  |/ ___ \| | 
 |_____|_|  |_|\___/ |_| |___|_| \_\_| \_/_/   \_\___|
```

# ✨ EmotionAI — Real-Time Facial Emotion Recognition & Analysis
### *Next-Generation Full-Stack Deep Learning & Live Facial Emotion Analysis Platform*

  <p align="center">
    <a href="#-overview">Overview</a> •
    <a href="#-key-highlights--core-capabilities">Key Features</a> •
    <a href="#-system-architecture">Architecture</a> •
    <a href="#-technology-stack">Tech Stack</a> •
    <a href="#-quick-start-guide">Quick Start</a> •
    <a href="#-project-structure">Project Structure</a>
  </p>

  <br />

  [![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
  [![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
  [![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

  <br />
  <hr />

</div>

<br />

## 🌟 Overview

**EmotionAI** is an enterprise-ready, full-stack computer vision and deep learning platform engineered for **real-time facial emotion recognition, affective computing, and video stream emotion analytics**.

Powered by a lightweight, high-performance **Mini-Xception Convolutional Neural Network (CNN)** trained on the **FER2013 dataset** (35,000+ facial images), EmotionAI achieves real-time inference latencies of **10–15ms on CPU** and **3–5ms on GPU**. Combined with an **OpenCV face-detection pipeline** and a production-grade **Flask backend**, the platform seamlessly delivers live webcam tracking and photo upload analysis through a modern **glassmorphism web interface**.

<br />

---

## ⚡ Key Highlights & Core Capabilities

### 🤖 Deep Learning & Mini-Xception CNN Engine
- **Ultra-Efficient Architecture**: Utilizes Depthwise Separable Convolutions, Residual Skip-Connections, and Batch Normalization to achieve high classification accuracy with only **~60,000 trainable parameters**.
- **Low-Latency Inference**: Optimized for real-time edge and web deployment without requiring expensive dedicated GPU hardware.
- **Robust Feature Extraction**: Trained to extract scale-invariant and pose-tolerant facial facial landmarks.

---

### 📷 Dual Emotion Detection Modes
- **📹 Real-Time Live Webcam Stream**:
  - Continuous webcam video processing via Canvas API and OpenCV.
  - Dynamic facial bounding box overlay with real-time emotion label tag and confidence percentage.
- **🖼️ High-Precision Image Upload Analysis**:
  - Multi-format image analysis supporting JPG, PNG, WEBP, and AVIF.
  - Interactive emotion confidence meters rendering multi-class probability distributions.

---

### 🎭 7-Class Affective Taxonomy
Classifies facial expressions into seven standardized psychological emotion categories:
- 😡 **Angry** | 🤢 **Disgust** | 😨 **Fear** | 😀 **Happy** | 😐 **Neutral** | 😢 **Sad** | 😲 **Surprise**

---

### 🎨 Modern Glassmorphism UI & UX
- **Sleek Aesthetic**: Translucent glassmorphic cards, vibrant accent gradients, dynamic micro-interactions, and dark mode styling.
- **Fully Responsive**: Fluid layout adapted seamlessly across desktop, tablet, and mobile browsers.

---

### 🏋️ Complete Model Training & Evaluation Pipeline
- **End-to-End Pipeline (`train_model.py`)**: Includes real-time image data augmentation, class-weight balancing to resolve dataset imbalance, and callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`).
- **Automated Performance Artifacts**: Exports evaluation plots directly to `training_results/` including confusion matrices and training history curves.

<br />

---

## 🏗️ System Architecture

```text
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                            User Web Browser                               │
 │   ┌───────────────────────────┐           ┌───────────────────────────┐   │
 │   │  Live Camera Stream (JS)  │           │   Photo Upload UI (HTML)  │   │
 │   └─────────────┬─────────────┘           └─────────────┬─────────────┘   │
 └─────────────────┼───────────────────────────────────────┼─────────────────┘
                   │ HTTP POST (Base64/Form)               │ HTTP POST Image
                   ▼                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                           Flask Backend (app.py)                          │
 │                                                                           │
 │   1. Image Decoding & Frame Capture                                       │
 │   2. Face Detection via OpenCV Haar Cascade / MTCNN                       │
 │   3. ROI Crop, Grayscale Conversion & Resize (48x48)                     │
 │   4. Tensor Normalization (X / 255.0)                                     │
 │   5. Keras Mini-Xception CNN Model Inference                              │
 │   6. Softmax Emotion Class Probabilities Calculation                      │
 └─────────────────┬─────────────────────────────────────────────────────────┘
                   │
                   ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                            JSON / Rendered Output                         │
 │  - Predicted Emotion Label & Confidence Score (%)                         │
 │  - Annotated Image / Live Bounding Box Stream Frame                       │
 └───────────────────────────────────────────────────────────────────────────┘
```

<br />

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | Python 3.8+ & Flask | Production REST server & endpoint routing |
| **Machine Learning** | TensorFlow 2.x & Keras | Mini-Xception CNN model training & inference |
| **Computer Vision** | OpenCV (`cv2`) | Face detection, Haar Cascade, frame transformation |
| **Data Science** | NumPy, Scikit-learn, Matplotlib | Matrix math, class weighting, evaluation metrics |
| **Frontend UI** | HTML5, Vanilla CSS3, JavaScript | Glassmorphism design system & webcam Media API |

<br />

---

## 🚀 Quick Start Guide

### 📋 Prerequisites
- **Python 3.8+** installed on your system.
- **Git** for cloning the repository.
- A functional **webcam** (for live emotion detection mode).

---

### 💻 Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/karmaboy1309/emotion-ai.git
   cd emotion-ai
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

### 🏃 Running the Application

Launch the Flask development server:
```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

- Visit `/` to test **Image Upload Mode**.
- Visit `/live` to open **Real-Time Live Webcam Detection Mode**.

---

### 🎯 Training the Model (Optional)

To retrain the Mini-Xception CNN on the FER2013 dataset:
```bash
python train_model.py
```
*Trained model weights will be saved to `models/emotion_model.keras` and analytics exported to `training_results/`.*

<br />

---

## 📁 Project Structure

```text
facial-emotion-recognation-fullstack-main/
├── dataset/                    # FER2013 dataset (train & test subsets)
│   └── fer2013/
├── models/                     # Trained TensorFlow/Keras models
│   └── emotion_model.keras     # Mini-Xception trained model weights
├── notebooks/                  # Jupyter notebooks for model experiments
│   └── FacialEmotion-Recognation.ipynb
├── static/                     # Web static assets
│   ├── css/                    # Custom Glassmorphism stylesheet (style.css)
│   └── js/                     # Client-side scripts & webcam handler
├── templates/                  # HTML Jinja templates
│   ├── index.html              # Photo upload home page
│   ├── live.html               # Real-time webcam emotion detection page
│   ├── about.html              # System architecture & model info
│   └── contact.html            # Contact & developer details
├── training_results/           # Exported metrics & confusion matrices
├── app.py                      # Core Flask web server & inference routes
├── train_model.py              # CNN model training script & data pipeline
├── haarcascade_frontalface_default.xml # OpenCV Haar Cascade face detector
├── requirements.txt            # Python dependencies manifest
├── ANALYSIS_SUMMARY.md         # Technical metrics summary
├── PROJECT_ANALYSIS.md         # Comprehensive project evaluation report
├── README.md                   # Project documentation
└── .gitignore                  # Security & repository protection rules
```

<br />

---

## 📄 License & Contact

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Developed with ❤️ by **[Anurag Pandey](https://github.com/karmaboy1309)**.
- 📧 Email: `anurag077269@gmail.com`
- 💼 LinkedIn: [Anurag Pandey](https://www.linkedin.com/in/anurag-pandey-704479253/)
- 🐙 GitHub: [@karmaboy1309](https://github.com/karmaboy1309)
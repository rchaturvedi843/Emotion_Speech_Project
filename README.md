# Emotion AI - Speech Emotion Recognition Project

This project leverages Machine Learning (SVM) to detect human emotions from speech audio. It features a beautiful, dynamic, and glassmorphism-styled web interface built with Vanilla CSS/JS and served using FastAPI.

## Features

- **High-Accuracy Emotion Detection**: Recognizes emotions like Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised.
- **FastAPI Backend**: A lightweight, lightning-fast Python server that extracts MFCC features and predicts emotions in real-time.
- **Modern UI**: A responsive, premium dark-mode interface with animated blobs and smooth user experiences.

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sudhir-chaturvedi/Emotion_Speech_Project.git
   cd Emotion_Speech_Project
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **IMPORTANT**: Ensure the `RAVDESS` dataset is placed inside `dataset/RAVDESS/` before training.

4. Train the Model:
   ```bash
   python code/emotion_recognition.py
   ```
   This will train the SVM model and save `svm_model.pkl` and `label_encoder.pkl` in the `outputs/` folder.

5. Run the Web App:
   ```bash
   python app.py
   ```
   Open `http://localhost:8000` in your web browser.

## Technologies Used
- Python (FastAPI, Librosa, Scikit-learn)
- HTML5, Vanilla CSS, Vanilla JS

## Author
Developed by Sudhir Chaturvedi.

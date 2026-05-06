# =========================================
# Emotion Recognition from Speech (ML)
# AUTO PATH VERSION - FINAL
# =========================================

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# -----------------------------------------
# AUTO PATH DETECTION (NO ../ NO ERROR)
# -----------------------------------------
CURRENT_FILE = os.path.abspath(__file__)
CODE_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_DIR = os.path.dirname(CODE_DIR)

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset", "RAVDESS")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

print("PROJECT DIR:", PROJECT_DIR)
print("DATASET DIR:", DATASET_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------
# STOP IF DATASET NOT FOUND
# -----------------------------------------
if not os.path.exists(DATASET_DIR):
    raise Exception(
        f"❌ Dataset folder NOT FOUND at:\n{DATASET_DIR}\n\n"
        "✔ Fix this by placing .wav files inside:\n"
        "Emotion_Speech_Project/dataset/RAVDESS/"
    )

# -----------------------------------------
# Emotion Mapping
# -----------------------------------------
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# -----------------------------------------
# MFCC Feature Extraction
# -----------------------------------------
def extract_mfcc(path):
    audio, sr = librosa.load(path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# -----------------------------------------
# Load Dataset
# -----------------------------------------
X, y = [], []

for file in os.listdir(DATASET_DIR):
    if file.endswith(".wav"):
        emotion_code = file.split("-")[2]
        emotion = emotion_map[emotion_code]

        features = extract_mfcc(os.path.join(DATASET_DIR, file))
        X.append(features)
        y.append(emotion)

X = np.array(X)
y = np.array(y)

# -----------------------------------------
# Encode Labels
# -----------------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------------------
# Train-Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------------------
# Train SVM
# -----------------------------------------
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# -----------------------------------------
# Predict & Evaluate
# -----------------------------------------
y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, target_names=le.classes_)
print("\nCLASSIFICATION REPORT\n")
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# -----------------------------------------
# Confusion Matrix
# -----------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title("Emotion Recognition Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
# plt.show() # Disable show so it doesn't block

# Save the model and label encoder
model_path = os.path.join(OUTPUT_DIR, "svm_model.pkl")
le_path = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
joblib.dump(model, model_path)
joblib.dump(le, le_path)

print(f"\n✅ Outputs and models saved in: {OUTPUT_DIR}\n")
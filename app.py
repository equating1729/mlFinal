import os
import shutil
import librosa
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import uvicorn

# ============================================================
# LOAD AND TRAIN MODEL
# ============================================================
print("Loading data and training model...")

df = pd.read_csv("features.csv")
X = df.drop("label", axis=1).values
y_raw = df["label"].values

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="rbf", C=10, gamma=0.01, random_state=42)
model.fit(X_train, y_train)

print("Model ready!")
print("Model expects feature count:", X.shape[1])
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).flat[0])

    feats = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [bandwidth_mean, bandwidth_std, tempo]
    ])

    return feats

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_genre_from_file(file_path):
    feats = extract_features(file_path)

    if len(feats) != X.shape[1]:
        raise ValueError(
            f"Extracted {len(feats)} features but model expects {X.shape[1]}"
        )

    feats_scaled = scaler.transform([feats])
    pred = model.predict(feats_scaled)[0]
    genre = le.inverse_transform([pred])[0]

    decision = model.decision_function(feats_scaled)[0]
    scores = {}

    for i, g in enumerate(le.classes_):
        scores[g] = float(decision[i])

    return genre.upper(), scores

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# static files (css/js/images)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_index():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        genre, scores = predict_genre_from_file(temp_path)

        return JSONResponse({
            "genre": genre,
            "scores": scores,
            "filename": file.filename
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=7860, reload=True)
import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "genres_original"

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    # 29 features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std  = np.std(bandwidth)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).flat[0])

    return np.concatenate([
        mfcc_mean, mfcc_std,
        [bandwidth_mean, bandwidth_std,
         tempo]
    ])

# Loop through all files
genres = os.listdir(DATASET_PATH)
rows = []
total = 0

for genre in genres:
    folder = os.path.join(DATASET_PATH, genre)
    files = os.listdir(folder)
    for fname in files:
        if fname.endswith('.wav'):
            path = os.path.join(folder, fname)
            try:
                feats = extract_features(path)
                rows.append([*feats, genre])
                total += 1
                print(f"Done: {total} - {fname}")
            except Exception as e:
                print(f"Skipped {fname}: {e}")

# Save to CSV
cols = [f"feat_{i}" for i in range(29)] + ["label"]
df = pd.DataFrame(rows, columns=cols)
df.to_csv("features.csv", index=False)

print("\nFinished!")
print(f"Total files processed: {len(df)}")
print(df.shape)
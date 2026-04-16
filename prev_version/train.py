import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# STEP 1 - LOAD DATA
# ============================================================
df = pd.read_csv("features.csv")
print("Dataset shape:", df.shape)
print("Genres:", df["label"].unique())
print("Samples per genre:\n", df["label"].value_counts())

# ============================================================
# STEP 2 - SEPARATE FEATURES AND LABELS
# ============================================================
X = df.drop("label", axis=1).values
y_raw = df["label"].values

# ============================================================
# STEP 3 - ENCODE LABELS
# ============================================================
le = LabelEncoder()
y = le.fit_transform(y_raw)

print("\nLabel mapping:")
for i, genre in enumerate(le.classes_):
    print(f"  {i} = {genre}")

# ============================================================
# STEP 4 - TRAIN TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# ============================================================
# STEP 5 - SCALE FEATURES
# ============================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("\nPreprocessing done!")
print("Mean of first feature after scaling:", round(X_train[:, 0].mean(), 4))

# ============================================================
# STEP 6 - TRAIN RBF CLASSIFIER
# ============================================================
model = SVC(kernel='rbf', C=10, gamma=0.01, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"\nAccuracy: {accuracy:.3f}")

# ============================================================
# STEP 7 - EVALUATION
# ============================================================
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ============================================================
# STEP 8 - CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap='Blues')
plt.ylabel('True Genre')
plt.xlabel('Predicted Genre')
plt.title('Confusion Matrix - RBF Classifier on GTZAN')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("\nConfusion matrix saved!")
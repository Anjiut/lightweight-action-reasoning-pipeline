import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Load pose feature dataset
data = np.load("pose_dataset.npz")
X = data["X"]   # shape: (N, 51)
y = data["y"]   # shape: (N,)

print("[INFO] Loaded dataset")
print("[INFO] X shape:", X.shape)
print("[INFO] y shape:", y.shape)

# Standardize features (fit on the full dataset)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train MLP classifier on pose features
classifier = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    max_iter=500,
    random_state=42
)
classifier.fit(X_scaled, y)

# Sanity check: performance on training set
print("\n[INFO] Training accuracy:")
print(classifier.score(X_scaled, y))

print("\n[INFO] Classification report (training set):")
y_pred = classifier.predict(X_scaled)
print(classification_report(y, y_pred))

# Save trained model and scaler for inference
joblib.dump(classifier, "mlp_action_model.pkl")
joblib.dump(scaler, "pose_scaler.pkl")

print("\n[INFO] Model saved as mlp_action_model.pkl")
print("[INFO] Scaler saved as pose_scaler.pkl")

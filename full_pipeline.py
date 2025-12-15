import os
import json
import joblib
import numpy as np

from build_dataset import keypoints_to_vector
from reasoning_agent import llm_reasoning, llm_temporal_reasoning

# ------------------------------------------------------------
# Paths and configuration
# ------------------------------------------------------------

# Directory containing pose keypoints extracted from new/test videos
NEW_POSE_DIR = "new_pose_keypoints"

# Trained MLP model and feature scaler
MODEL_PATH = "mlp_action_model.pkl"
SCALER_PATH = "pose_scaler.pkl"

# Action label mapping (must match training order)
ACTIONS = ["open_door", "pick_book", "pour_water", "walk_stop"]
IDX_TO_LABEL = {idx: label for idx, label in enumerate(ACTIONS)}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def classify_video(action_name: str, model, scaler):
    """
    Classify a video by aggregating frame-level predictions.

    Args:
        action_name: Name of the action folder under NEW_POSE_DIR.
        model: Trained MLP classifier.
        scaler: Fitted StandardScaler.

    Returns:
        majority_label (str): Predicted action label for the video.
        counts (dict): Frame-level prediction counts.
    """
    pose_dir = os.path.join(NEW_POSE_DIR, action_name)
    if not os.path.isdir(pose_dir):
        print(f"[WARNING] Directory not found: {pose_dir}")
        return None, {}

    json_files = sorted(
        f for f in os.listdir(pose_dir)
        if f.endswith(".json") and not f.startswith(".")
    )

    if not json_files:
        print(f"[WARNING] No pose files found in {pose_dir}")
        return None, {}

    features = []

    for fname in json_files:
        fpath = os.path.join(pose_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        keypoints = data.get("keypoints", None)

        # Convert keypoints to the same 51-D feature used during training
        feature_vec = keypoints_to_vector(keypoints, dim=51)
        features.append(feature_vec)

    X = np.stack(features, axis=0)        # shape: (num_frames, 51)
    X_scaled = scaler.transform(X)

    # Frame-level predictions
    frame_preds = model.predict(X_scaled)

    # Majority vote
    unique, counts = np.unique(frame_preds, return_counts=True)
    count_dict = {IDX_TO_LABEL[int(k)]: int(v) for k, v in zip(unique, counts)}

    majority_idx = int(unique[np.argmax(counts)])
    majority_label = IDX_TO_LABEL[majority_idx]

    return majority_label, count_dict

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def main():
    print("=== Human Action Reasoning Pipeline ===\n")

    # Load trained model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    action_history = []

    for action in ACTIONS:
        print(f"[INFO] Processing video for action folder: {action}")

        predicted_label, frame_counts = classify_video(action, model, scaler)

        if predicted_label is None:
            print("[WARNING] Skipping due to missing or invalid data.\n")
            continue

        action_history.append(predicted_label)

        print(f"[INFO] Majority prediction: {predicted_label}")
        print(f"[INFO] Frame-level counts: {frame_counts}\n")

        # Per-action reasoning
        reasoning = llm_reasoning(predicted_label)
        print("=== Per-action Reasoning ===")
        print(json.dumps(reasoning, indent=2))
        print("\n" + "-" * 50 + "\n")

    # Temporal reasoning over the full action sequence
    if action_history:
        print("=== Temporal Reasoning Over Action Sequence ===")
        temporal_result = llm_temporal_reasoning(action_history)
        print(json.dumps(temporal_result, indent=2))
        print("\n" + "=" * 70 + "\n")

    print("[INFO] Pipeline completed successfully.")

# ------------------------------------------------------------

if __name__ == "__main__":
    main()






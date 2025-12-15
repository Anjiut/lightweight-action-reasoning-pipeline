import os
import json
import numpy as np
import joblib
from collections import Counter

NEW_POSE_DIR = "new_pose_keypoints"
ACTIONS = ["open_door", "pick_book", "pour_water", "walk_stop"]
label_map = {i: a for i, a in enumerate(ACTIONS)}

def kp_to_vector(kps, dim=51):
    if kps is None:
        return np.zeros(dim, dtype=np.float32)
    arr = np.array(kps, dtype=np.float32).reshape(-1, 3)
    flat = arr.flatten()
    if flat.size >= dim:
        return flat[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[:flat.size] = flat
    return out

# 载入模型和 scaler
clf = joblib.load("mlp_action_model.pkl")
scaler = joblib.load("pose_scaler.pkl")

print("Testing on NEW videos...\n")

for action in ACTIONS:
    action_dir = os.path.join(NEW_POSE_DIR, action)
    if not os.path.isdir(action_dir):
        print(f"Warning: {action_dir} not found, skip")
        continue

    files = sorted(
        f for f in os.listdir(action_dir)
        if f.endswith(".json") and not f.startswith(".")
    )
    print(f"Processing new action video: {action} ({len(files)} frames)")

    X_frames = []
    for fname in files:
        fpath = os.path.join(action_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        kps = data.get("keypoints", None)
        vec = kp_to_vector(kps, dim=51)
        X_frames.append(vec)

    X_frames = np.stack(X_frames, axis=0)  # (T, 51)
    X_frames_scaled = scaler.transform(X_frames)

    preds = clf.predict(X_frames_scaled)   # frame-level 预测
    counts = Counter(preds)
    majority_label = counts.most_common(1)[0][0]
    majority_action = label_map[majority_label]

    print(f"  Majority prediction: {majority_action}")
    frame_label_counts = {label_map[k]: v for k, v in counts.items()}
    print("  Frame label counts:", frame_label_counts)

   

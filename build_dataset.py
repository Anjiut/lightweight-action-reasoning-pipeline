import os
import json
import numpy as np

# Directory containing pose keypoints extracted from training videos
POSE_DIR = "pose_keypoints"

# Predefined action classes
ACTIONS = ["open_door", "pick_book", "pour_water", "walk_stop"]

# Map action name to label index
LABEL_MAP = {action: idx for idx, action in enumerate(ACTIONS)}

def keypoints_to_vector(keypoints, dim=51):
    """
    Convert pose keypoints to a fixed-length feature vector.

    Args:
        keypoints: list of [x, y, confidence] with length ~17,
                   or None if no person is detected.
        dim: output feature dimension (default: 51).

    Returns:
        A numpy array of shape (dim,). If keypoints is None,
        returns a zero vector.
    """
    if keypoints is None:
        return np.zeros(dim, dtype=np.float32)

    arr = np.array(keypoints, dtype=np.float32).reshape(-1, 3)
    flat = arr.flatten()  # shape: (N * 3,)

    if flat.size >= dim:
        return flat[:dim]
    else:
        # Pad with zeros if fewer than dim values (unlikely in practice)
        output = np.zeros(dim, dtype=np.float32)
        output[:flat.size] = flat
        return output


X_list = []
y_list = []

for action in ACTIONS:
    action_dir = os.path.join(POSE_DIR, action)

    if not os.path.isdir(action_dir):
        print(f"[WARNING] Directory not found, skipping: {action_dir}")
        continue

    json_files = sorted(
        f for f in os.listdir(action_dir)
        if f.endswith(".json") and not f.startswith(".")
    )

    print(f"[INFO] Processing action '{action}' ({len(json_files)} frames)")

    for fname in json_files:
        fpath = os.path.join(action_dir, fname)

        with open(fpath, "r") as f:
            data = json.load(f)

        keypoints = data.get("keypoints", None)
        feature_vec = keypoints_to_vector(keypoints, dim=51)

        X_list.append(feature_vec)
        y_list.append(LABEL_MAP[action])

# Stack dataset
X = np.stack(X_list, axis=0)  # shape: (N, 51)
y = np.array(y_list, dtype=np.int64)

print("[INFO] Dataset built successfully")
print("[INFO] X shape:", X.shape)
print("[INFO] y shape:", y.shape)

# Save dataset
np.savez("pose_dataset.npz", X=X, y=y)
print("[INFO] Saved dataset to pose_dataset.npz")

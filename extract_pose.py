import os
import json
import numpy as np
from PIL import Image
import openpifpaf

# Directory of extracted frames (from training videos)
FRAME_DIR = "frames"

# Directory to save pose keypoints (training set)
POSE_DIR = "pose_keypoints"

os.makedirs(POSE_DIR, exist_ok=True)

# Initialize OpenPifPaf predictor with the default checkpoint
predictor = openpifpaf.Predictor()

# Only keep valid action subfolders (ignore hidden files like .DS_Store)
actions = [
    a for a in os.listdir(FRAME_DIR)
    if os.path.isdir(os.path.join(FRAME_DIR, a)) and not a.startswith(".")
]

print("[INFO] Extracting pose keypoints for TRAINING videos...")

for action in actions:
    print(f"\n[INFO] Processing action: {action}")
    action_dir = os.path.join(FRAME_DIR, action)

    output_action_dir = os.path.join(POSE_DIR, action)
    os.makedirs(output_action_dir, exist_ok=True)

    # Process only JPG frames
    frame_files = sorted(
        f for f in os.listdir(action_dir)
        if f.endswith(".jpg") and not f.startswith(".")
    )

    for frame_name in frame_files:
        frame_path = os.path.join(action_dir, frame_name)
        output_path = os.path.join(output_action_dir, frame_name.replace(".jpg", ".json"))

        # Load image with PIL and convert to numpy array (H, W, 3)
        pil_img = Image.open(frame_path).convert("RGB")
        np_img = np.asarray(pil_img)

        # OpenPifPaf inference (returns predictions, scores, fields)
        predictions, _, _ = predictor.numpy_image(np_img)

        if predictions:
            # Use the most prominent person only (first prediction)
            first_pred = predictions[0]
            # Shape ~ (17, 3): (x, y, confidence)
            keypoints = first_pred.data.tolist()
        else:
            keypoints = None

        with open(output_path, "w") as f:
            json.dump({"keypoints": keypoints}, f)

        print(f"[INFO] Saved: {output_path}")

print("\n[INFO] All poses extracted successfully for TRAINING videos!")

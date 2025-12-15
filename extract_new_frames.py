import os
import cv2

# Directory containing new/test videos
VIDEO_DIR = "new_videos"

# Directory to store extracted frames from new/test videos
FRAME_DIR = "new_frames"

os.makedirs(FRAME_DIR, exist_ok=True)

# Automatically load all mp4 files
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

print("[INFO] Extracting frames from NEW videos...")

for video_file in video_files:
    video_path = os.path.join(VIDEO_DIR, video_file)

    # Get video name without extension
    video_name = os.path.splitext(video_file)[0]

    # Normalize action name by removing optional 'new_' prefix
    if video_name.startswith("new_"):
        action_name = video_name[4:]
    else:
        action_name = video_name

    output_dir = os.path.join(FRAME_DIR, action_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Processing video: {video_file} -> action: {action_name}")

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    print(f"[INFO] Extracted {frame_idx} frames for action '{action_name}'")

print("\n[INFO] All new videos processed successfully.\n")

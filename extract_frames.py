import os
import cv2

# Directory containing input videos
VIDEO_DIR = "videos"

# Directory to store extracted frames
FRAME_DIR = "frames"

os.makedirs(FRAME_DIR, exist_ok=True)

# Automatically load all mp4 videos in the video directory
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

for video_name in video_files:
    video_path = os.path.join(VIDEO_DIR, video_name)

    # Use the video filename (without extension) as the action name
    action_name = os.path.splitext(video_name)[0]

    output_dir = os.path.join(FRAME_DIR, action_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Processing video: {video_name}")

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"{frame_idx:05d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        frame_idx += 1

    cap.release()
    print(f"[INFO] Extracted {frame_idx} frames for action '{action_name}'")

print("[INFO] All videos processed successfully.")

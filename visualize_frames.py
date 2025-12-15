import os
import random
import matplotlib.pyplot as plt
from PIL import Image

FRAME_DIR = "frames"

# 随机选择一个动作类别
actions = os.listdir(FRAME_DIR)
action = random.choice(actions)

action_path = os.path.join(FRAME_DIR, action)
frames = sorted(os.listdir(action_path))

# 随机选 9 张图
selected_frames = random.sample(frames[:50], 9)

plt.figure(figsize=(10, 10))
plt.suptitle(f"Action: {action}", fontsize=20)

for i, frame_name in enumerate(selected_frames):
    frame_path = os.path.join(action_path, frame_name)
    img = Image.open(frame_path)

    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis('off')

plt.tight_layout()
plt.show()

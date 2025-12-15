import json
import matplotlib.pyplot as plt
import numpy as np
import os

JSON_PATH = "pose_keypoints/open_door/00050.json"  # ← 你可以换成任何 JSON

# 关键点连接规则（COCO）
EDGES = [
    (5 ,7), (7 ,9),   # 左臂
    (6 ,8), (8 ,10),  # 右臂
    (11,13), (13,15), # 左腿
    (12,14), (14,16), # 右腿
    (5,6), (11,12),   # 肩、臀
    (5,11), (6,12)    # 身体连接
]

# 载入 JSON
with open(JSON_PATH) as f:
    data = json.load(f)

keypoints = data["keypoints"]
keypoints = np.array(keypoints).reshape(-1, 3)

# 绘制关键点
plt.figure(figsize=(5,8))
for x, y, conf in keypoints:
    if conf > 0.3:
        plt.scatter(x, -y, c="red")   # y 反转以符合视觉习惯

# 绘制骨架
for a, b in EDGES:
    x1, y1, c1 = keypoints[a]
    x2, y2, c2 = keypoints[b]
    if c1 > 0.3 and c2 > 0.3:
        plt.plot([x1, x2], [-y1, -y2], c="blue")

plt.title("Pose Visualization")
plt.axis("equal")
plt.show()

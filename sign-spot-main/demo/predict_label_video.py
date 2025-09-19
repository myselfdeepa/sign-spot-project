import json
import numpy as np
import torch
from utils import load_model, load_rgb_video, prepare_input, sliding_windows, viz_similarities
from pathlib import Path
import cv2
import os

# ===== Load label map from your word_index.json =====
label_path = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\bsldict\word_index.json")
with open(label_path, 'r') as f:
    label_map_data = json.load(f)
    label_map = [None] * len(label_map_data)
    for entry in label_map_data:
        word = entry["Word"]
        idx = entry["Index"]
        label_map[idx] = word

# ===== Constants =====
CHECKPOINT_PATH = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\models\i3d_mlp.pth.tar")
#CHECKPOINT_PATH = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\models\i3d_mlp.pth")
VIDEO_PATH = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\demo\sample_data\input.mp4")
OUTPUT_VIDEO_PATH = Path(__file__).parent / "output_with_labels.mp4"
ARCH = 'i3d_mlp'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# ===== Load model =====
model = load_model(checkpoint_path=CHECKPOINT_PATH, arch=ARCH)
model.eval()

# ===== Load video =====
rgb_video = load_rgb_video(VIDEO_PATH, fps=25)
if isinstance(rgb_video, list):
    rgb_video = [torch.from_numpy(f) if not isinstance(f, torch.Tensor) else f for f in rgb_video]
    rgb_video = torch.stack(rgb_video, dim=1)

print("✅ rgb_video shape:", rgb_video.shape)

# ===== Prepare input and run predictions =====
rgb_input = prepare_input(rgb_video)
rgb_slides, t_mids = sliding_windows(rgb=rgb_input, stride=1, num_in_frames=16)

predicted_labels = []
with torch.no_grad():
    for segment in rgb_slides:
        segment = segment.unsqueeze(0)
        outputs = model(segment)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        predicted_labels.append(predicted_idx)

# ===== Create similarity array =====
sim_array = np.zeros((len(predicted_labels), len(label_map)), dtype=np.float32)
for i, idx in enumerate(predicted_labels):
    sim_array[i][idx] = 1.0

# ===== Visualize predictions =====
viz_similarities.extra_args = {
    "labels": label_map,
    "codec": fourcc
}
viz_similarities(
    rgb_video,
    t_mids,
    sim_array,
    0.5,
    keyword="demo",
    output_path=OUTPUT_VIDEO_PATH,
    viz_with_dict=False,
    dict_video_links=([], []),
    
)

# ===== Confirm and open output =====
print("✅ Labeled output video created at:", OUTPUT_VIDEO_PATH)
if OUTPUT_VIDEO_PATH.exists():
    os.startfile(OUTPUT_VIDEO_PATH)
else:
    print("❌ Output video was not generated.")

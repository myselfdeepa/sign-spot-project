import torch
import torch.nn as nn
from utils import load_rgb_video, prepare_input, load_model, sliding_windows
from pathlib import Path

class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(FeatureEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        # Flatten the features before passing through the linear layer
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)

def extract_features_from_continuous(video_path, model, embedding_model, stride=1, num_in_frames=16):
    # Load and preprocess video
    video_tensor = load_rgb_video(video_path, fps=25)
    processed_video = prepare_input(video_tensor)

    # Apply sliding windows
    rgb_slides, _ = sliding_windows(rgb=processed_video, stride=stride, num_in_frames=num_in_frames)

    # Extract and transform features from each clip
    features_list = []
    with torch.no_grad():
        for clip in rgb_slides:
            features = model(clip.unsqueeze(0))['embds']
            embedded_features = embedding_model(features)
            features_list.append(embedded_features)

    return torch.stack(features_list)

if __name__ == "__main__":
    checkpoint_path = Path("/home/uday/Desktop/SignSpotting/bsldict/models/i3d.pth.tar")
    video_path = Path("/home/uday/Desktop/SignSpotting/bsldict/demo/sample_data/input.mp4")

    # Load models
    model = load_model(checkpoint_path, arch='i3d')
    model.eval()
    embedding_model = FeatureEmbedding(input_dim=1024, embedding_dim=256)  # Adjust dimensions as needed

    # Extract and transform features
    features = extract_features_from_continuous(video_path, model, embedding_model)
    print("Extracted Embedded Features:", features.shape)


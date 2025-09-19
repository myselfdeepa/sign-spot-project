"""
import torch
from pathlib import Path
from utils import load_rgb_video, prepare_input, load_model# Replace 'your_module' with the actual module name where these functions/classes are defined.

def extract_features_from_isolated(video_path, model):
    # Load and preprocess video
    video_tensor = load_rgb_video(video_path, fps=25)
    processed_video = prepare_input(video_tensor)

    # Extract features using I3D model
    with torch.no_grad():
        features = model(processed_video.unsqueeze(0))['embds']
    return features

if __name__ == "__main__":
    checkpoint_path = Path("/home/uday/Desktop/SignSpotting/bsldict/models/i3d.pth.tar")  # Replace with your model's checkpoint path
    video_path = Path("/home/uday/Desktop/SignSpotting/bsldict/bsldict/videos_original/a_001_004_000_Aaronic.mp4")  # Replace with your video file path

    # Load I3D model
    model = load_model(checkpoint_path, arch='i3d')  # Ensure the architecture key is correct
    model.eval()

    # Extract features
    features = extract_features_from_isolated(video_path, model)
    print("Extracted Features:", features.shape)
"""

import torch
import torch.nn as nn
from pathlib import Path
from utils import load_rgb_video, prepare_input, load_model

class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(FeatureEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        # Flatten the tensor from [1, 1024, 8, 1, 1] to [1, 8192]
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)

def extract_features_from_isolated(video_path, model, embedding_model):
    # Load and preprocess video
    video_tensor = load_rgb_video(video_path, fps=25)
    processed_video = prepare_input(video_tensor)

    # Extract features using I3D model
    with torch.no_grad():
        features = model(processed_video.unsqueeze(0))['embds']

    # Flatten the features and apply the linear embedding layer
    flattened_features = features.view(1, -1)
    embedded_features = embedding_model(flattened_features)
    return embedded_features


if __name__ == "__main__":
    checkpoint_path = Path("/home/uday/Desktop/SignSpotting/bsldict/models/i3d.pth.tar")  # Replace with your model's checkpoint path
    video_path = Path("/home/uday/Desktop/SignSpotting/bsldict/bsldict/videos_original/a_001_004_000_Aaronic.mp4")  # Replace with your video file path

    # Load I3D model
    model = load_model(checkpoint_path, arch='i3d')  # Ensure the architecture key is correct
    model.eval()

    # Initialize the linear embedding model
    embedding_model = FeatureEmbedding(input_dim=8192, embedding_dim=256)  # Adjust embedding_dim as needed

    # Extract features and process through embedding layer
    embedded_features = extract_features_from_isolated(video_path, model, embedding_model)
    print("Embedded Features:", embedded_features.shape)


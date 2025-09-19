"""
working
import torch
import math
from pathlib import Path
from utils import load_rgb_video, prepare_input, load_model, sliding_windows

def positional_encoding(dim, sentence_length):
    pe = torch.zeros(sentence_length, dim)
    for pos in range(sentence_length):
        for i in range(0, dim, 2):
            pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/dim)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/dim)))
    return pe
    
def extract_features_from_continuous(video_path, model, stride=1, num_in_frames=16, embedding_size=1024):
    # Load and preprocess video
    video_tensor = load_rgb_video(video_path, fps=25)
    processed_video = prepare_input(video_tensor)

    # Apply sliding windows
    rgb_slides, _ = sliding_windows(rgb=processed_video, stride=stride, num_in_frames=num_in_frames)

    # Positional Encoding
    pe = positional_encoding(embedding_size, len(rgb_slides))

    # Extract features from each clip
    features_list = []
    with torch.no_grad():
        for idx, clip in enumerate(rgb_slides):
            features = model(clip.unsqueeze(0))['embds']
            # Reshape features to [num_in_frames, 1024]
            features = features.squeeze().view(-1, 1024)
            # Add positional encoding
            features += pe[idx]
            features_list.append(features)

    # Stack the list of tensors into a 2D tensor
    features_2d = torch.stack(features_list).squeeze()
    return features_2d

if __name__ == "__main__":
    checkpoint_path = Path("/home/uday/Desktop/SignSpotting/bsldict/models/i3d.pth.tar")
    video_path = Path("/home/uday/Desktop/SignSpotting/bsldict/demo/sample_data/input.mp4")

    # Load I3D model
    model = load_model(checkpoint_path, arch='i3d')
    model.eval()

    # Extract and transform features
    features = extract_features_from_continuous(video_path, model)
    print("Features Shape:", features.shape)

"""

"""pos encoding
import torch
import torch.nn as nn
import math
from pathlib import Path
from utils import load_rgb_video, prepare_input, load_model, sliding_windows

class FeatureTransformer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeatureTransformer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

def positional_encoding(dim, sentence_length):
    pe = torch.zeros(sentence_length, dim)
    for pos in range(sentence_length):
        for i in range(0, dim, 2):
            pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/dim)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/dim)))
    return pe
    
def extract_features_from_continuous(video_path, model, transformer, stride=1, num_in_frames=16, embedding_size=512, print_samples=3):
    # Load and preprocess video
    video_tensor = load_rgb_video(video_path, fps=25)
    processed_video = prepare_input(video_tensor)

    # Apply sliding windows
    rgb_slides, _ = sliding_windows(rgb=processed_video, stride=stride, num_in_frames=num_in_frames)

    # Positional Encoding
    pe = positional_encoding(embedding_size, len(rgb_slides))

    # Extract features from each clip
    features_list = []
    with torch.no_grad():
        for idx, clip in enumerate(rgb_slides):
            features = model(clip.unsqueeze(0))['embds']
            # Reshape and apply linear transformation
            transformed_features = transformer(features.squeeze().view(-1, 1024))
            # Add positional encoding
            transformed_features += pe[idx]
            features_list.append(transformed_features)

            # Print positional encoding and transformed features for first few samples
          #  if idx < print_samples:
             #   print(f"Positional Encoding for position {idx}: \n{pe[idx]}")
               # print(f"Transformed Feature with Positional Encoding at position {idx}: \n{features_list[idx]}")

    # Stack the list of tensors into a 2D tensor
    features_2d = torch.stack(features_list).squeeze()
    return features_2d


if __name__ == "__main__":
    checkpoint_path = Path("/home/uday/Desktop/SignSpotting/bsldict/models/i3d.pth.tar")
    video_path = Path("/home/uday/Desktop/SignSpotting/bsldict/demo/sample_data/input.mp4")

    # Load I3D model
    model = load_model(checkpoint_path, arch='i3d')
    model.eval()

    # Define the feature transformer with the desired output size
    embedding_size = 512  # or any other size you need for your Transformer model
    transformer = FeatureTransformer(1024, embedding_size)

    # Extract and transform features
    transformed_features = extract_features_from_continuous(video_path, model, transformer, embedding_size=embedding_size)
    print("Transformed Features Shape:", transformed_features.shape)


"""

import torch
import torch.nn as nn
from pathlib import Path
from utils import load_rgb_video, prepare_input, load_model, sliding_windows

def extract_features_from_continuous(video_path, model, stride=1, num_in_frames=16):
    # Load and preprocess video
    video_tensor = load_rgb_video(video_path, fps=25)
    processed_video = prepare_input(video_tensor)

    # Apply sliding windows
    rgb_slides, _ = sliding_windows(rgb=processed_video, stride=stride, num_in_frames=num_in_frames)

    # Extract features from each clip
    features_list = []
    with torch.no_grad():
        for clip in rgb_slides:
            features = model(clip.unsqueeze(0))['embds'].squeeze()
            features_list.append(features)

    # Stack the list of tensors into a 2D tensor
    features_2d = torch.stack(features_list)
    return features_2d

if __name__ == "__main__":
    checkpoint_path = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\models\i3d.pth.tar")
    video_path = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\demo\sample_data/input.mp4")

    # Load I3D model
    model = load_model(checkpoint_path, arch='i3d')
    model.eval()

    # Extract and transform features
    features = extract_features_from_continuous(video_path, model)
    print("Features Shape:", features.shape)
    
def run_cfe():
    checkpoint_path = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\models\i3d.pth.tar")
    video_path = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\demo\sample_data/input.mp4")

    # Load I3D model
    model = load_model(checkpoint_path, arch='i3d')
    model.eval()

    # Extract and transform features
    features = extract_features_from_continuous(video_path, model)
    print("Features Shape:", features.shape)
    return features
    


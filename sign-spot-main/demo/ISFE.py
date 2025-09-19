import torch
import torch.nn as nn
from pathlib import Path
from utils import load_rgb_video, prepare_input, load_model

def extract_features_from_isolated(video_path, model):
    # Load and preprocess video
    video_tensor = load_rgb_video(video_path, fps=25)
    processed_video = prepare_input(video_tensor)

    # Extract features using I3D model
    with torch.no_grad():
        features = model(processed_video.unsqueeze(0))['embds']
        # Reshape features to [8, 1024]
        features = features.squeeze().view(-1, 1024)
        return features

def run_isfe():
    #checkpoint_path = Path("/home/uday/Desktop/SignSpotting/bsldict/models/i3d.pth.tar")
    #video_path = Path("/home/uday/Desktop/SignSpotting/bsldict/bsldict/videos_original/t_010_030_001_trees.mp4")
    checkpoint_path = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\models\i3d.pth.tar")
    video_path = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\bsldict\videos_original\apple_25fps.mp4")

    # Load I3D model
    model = load_model(checkpoint_path, arch='i3d')
    model.eval()

    # Extract features
    features = extract_features_from_isolated(video_path, model)
    print("Features Shape:", features.shape)
    return features

if __name__ == "__main__":
    checkpoint_path = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\models\i3d.pth.tar")
    video_path = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\bsldict\videos_original\apple_25fps.mp4")

    # Load I3D model
    model = load_model(checkpoint_path, arch='i3d')
    model.eval()

    # Extract features
    features = extract_features_from_isolated(video_path, model)
    print("Features Shape:", features.shape)
    



"""
Was removing the 1024 to 512 code
import torch
import torch.nn as nn
from pathlib import Path
from utils import load_rgb_video, prepare_input, load_model


class FeatureTransformer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeatureTransformer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def extract_features_from_isolated(video_path, model:
    # Load and preprocess video
    video_tensor = load_rgb_video(video_path, fps=25)
    processed_video = prepare_input(video_tensor)

    # Extract features using I3D model
    with torch.no_grad():
        features = model(processed_video.unsqueeze(0))['embds']
        # Reshape features to [8, 1024]
        features = features.squeeze().view(-1, 1024)
        # Apply linear transformation
        #transformed_features = transformer(features)
        return transformed_features

if __name__ == "__main__":
    checkpoint_path = Path("/home/uday/Desktop/SignSpotting/bsldict/models/i3d.pth.tar")
    video_path = Path("/home/uday/Desktop/SignSpotting/bsldict/bsldict/videos_original/a_001_004_000_Aaronic.mp4")

    # Load I3D model
    model = load_model(checkpoint_path, arch='i3d')
    model.eval()

    # Define the feature transformer with the desired output size
    #embedding_size = 512  # or any other size you need for your Transformer model
    #transformer = FeatureTransformer(1024, embedding_size)

    # Extract and transform features
    features = extract_features_from_isolated(video_path, model)
    print("Transformed Features Shape:", features.shape)


"""





"""import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from utils import load_rgb_video, prepare_input, load_model

class FeatureTransformer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeatureTransformer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

def get_positional_encoding(max_seq_len, embed_size):
    pos_enc = torch.zeros(max_seq_len, embed_size)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-np.log(10000.0) / embed_size))
    
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)

    return pos_enc

def extract_features_from_isolated(video_path, model, transformer, max_seq_len, embed_size):
    # Load and preprocess video
    video_tensor = load_rgb_video(video_path, fps=25)
    processed_video = prepare_input(video_tensor)

    # Extract features using I3D model
    with torch.no_grad():
        features = model(processed_video.unsqueeze(0))['embds']
        # Reshape features to [8, 1024]
        features = features.squeeze().view(-1, 1024)
        # Apply linear transformation
        transformed_features = transformer(features)

        # Add positional encoding
        pos_encoding = get_positional_encoding(max_seq_len, embed_size)
        transformed_features_with_pos = transformed_features + pos_encoding[:transformed_features.size(0), :]
        
        return transformed_features_with_pos

if __name__ == "__main__":
    checkpoint_path = Path("/home/uday/Desktop/SignSpotting/bsldict/models/i3d.pth.tar")
    video_path = Path("/home/uday/Desktop/SignSpotting/bsldict/bsldict/videos_original/a_001_004_000_Aaronic.mp4")

    # Load I3D model
    model = load_model(checkpoint_path, arch='i3d')
    model.eval()

    # Define the feature transformer with the desired output size
    embedding_size = 512  # or any other size you need for your Transformer model
    max_seq_len = 8  # Maximum sequence length for positional encoding
    transformer = FeatureTransformer(1024, embedding_size)

    # Extract, transform, and add positional encoding to features
    features_with_pos_encoding = extract_features_from_isolated(video_path, model, transformer, max_seq_len, embedding_size)
    print("Features with Positional Encoding Shape:", features_with_pos_encoding.shape)
"""

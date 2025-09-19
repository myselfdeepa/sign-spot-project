import torch
import torch.nn as nn
from ISFE import extract_features_from_isolated
from CFE import extract_features_from_continuous
from pathlib import Path
from utils import load_model

class SignSpottingTransformer(nn.Module):
    def __init__(self, feature_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout_rate=0.1):
        super(SignSpottingTransformer, self).__init__()
        # Encoder and Decoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, isolated_features, continuous_features):
        # Pass the isolated features through the encoder
        encoded_features = self.transformer_encoder(isolated_features)

        # Pass the continuous features and encoded isolated features through the decoder
        decoded_features = self.transformer_decoder(continuous_features, encoded_features)

        return decoded_features
        
def run_sign_spotter():
    # Paths and model parameters
    #checkpoint_path = Path("/home/uday/Desktop/SignSpotting/bsldict/models/i3d.pth.tar")
    #video_path_isolated = Path("/home/uday/Desktop/SignSpotting/bsldict/bsldict/videos_original/a_009_042_000_apple.mp4")
    #video_path_continuous = Path("/home/uday/Desktop/SignSpotting/bsldict/demo/sample_data/input.mp4")
    checkpoint_path = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\models\i3d.pth.tar")
    video_path_isolated = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\bsldict\videos_original\apple_25fps.mp4")
    video_path_continuous = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\demo\sample_data/input.mp4")


    # Load I3D model
    model = load_model(checkpoint_path, arch='i3d')
    model.eval()

    # Extract and transform features for both isolated and continuous videos
    isolated_features = extract_features_from_isolated(video_path_isolated, model)
    continuous_features = extract_features_from_continuous(video_path_continuous, model)

    # SignSpottingTransformer parameters
    feature_dim = 1024  # Adjusted to match the I3D output dimension
    num_heads = 4  # Number of heads in multi-head attention
    num_encoder_layers = 4  # Number of encoder layers
    num_decoder_layers = 4  # Number of decoder layers

    # Instantiate the SignSpottingTransformer model
    sign_spotting_transformer = SignSpottingTransformer(feature_dim, num_heads, num_encoder_layers, num_decoder_layers)

    # Process features through the transformer
    output_features = sign_spotting_transformer(isolated_features.unsqueeze(1), continuous_features.unsqueeze(1))

    return output_features
    
if __name__ == "__main__":
    # Paths and model parameters
    checkpoint_path = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\models\i3d.pth.tar")
    video_path_isolated = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\bsldict\videos_original\apple_25fps.mp4")
    video_path_continuous = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\demo\sample_data\input.mp4")

    # Load I3D model
    model = load_model(checkpoint_path, arch='i3d')
    model.eval()

    # Extract and transform features for both isolated and continuous videos
    isolated_features = extract_features_from_isolated(video_path_isolated, model)
    continuous_features = extract_features_from_continuous(video_path_continuous, model)

    print("Isolated Features Shape:", isolated_features.shape)
    print("Continuous Features Shape:", continuous_features.shape)
    # SignSpottingTransformer parameters
    feature_dim = 1024  # Adjusted to match the I3D output dimension
    num_heads = 4  # Number of heads in multi-head attention
    num_encoder_layers = 1  # Number of encoder layers
    num_decoder_layers = 1  # Number of decoder layers

    # Instantiate the SignSpottingTransformer model
    sign_spotting_transformer = SignSpottingTransformer(feature_dim, num_heads, num_encoder_layers, num_decoder_layers)

    # Process features through the transformer
    output_features = sign_spotting_transformer(isolated_features.unsqueeze(1), continuous_features.unsqueeze(1))

    print("Output shape:", output_features.shape)  # Expected shape: [seq_len, 1, feature_dim]


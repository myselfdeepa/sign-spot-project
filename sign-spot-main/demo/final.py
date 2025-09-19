import torch
from utils import load_model, load_rgb_video, prepare_input, sliding_windows
from pathlib import Path

# Constants
CHECKPOINT_PATH = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\models\i3d_mlp.pth.tar")

VIDEO_PATH = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main\sign-spot-main\demo\sample_data/input.mp4")
ARCH = 'i3d_mlp'  # Architecture type

def main():
    # Load the combined i3d_mlp model with pre-trained weights
    model = load_model(checkpoint_path=CHECKPOINT_PATH, arch=ARCH)

    # Load and preprocess the input video
    rgb_video = load_rgb_video(VIDEO_PATH, fps=25)
    rgb_input = prepare_input(rgb_video)

    # Apply sliding windows to the input video
    rgb_slides, t_mid = sliding_windows(rgb=rgb_input, stride=1, num_in_frames=16)

    # Process each segment through the model and find the class with the highest probability
    for segment in rgb_slides:
        segment = segment.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(segment)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
            
            # Find the class index with the highest probability for each segment
            highest_prob_class_indices = torch.argmax(probabilities, dim=1)
            print("Highest Probability Class Index:", highest_prob_class_indices.item())

            # Optional: You can also check the value of the highest probability
            highest_prob_value = probabilities[0, highest_prob_class_indices.item()]
            print("Highest Probability Value:", highest_prob_value.item())

    # Optional: Visualization
    # You can use viz_similarities or other utility functions for visualization
    # ...

if __name__ == "__main__":
    main()

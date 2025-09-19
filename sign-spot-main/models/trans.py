import torch
from torchvision.io import read_video

# Load the pre-trained InceptionI3d model
from i3d import InceptionI3d  # Replace 'your_module' with the actual module where InceptionI3d is defined

# Function to load and preprocess a video
def load_and_preprocess_video(video_path):
    # Read video frames
    video, audio, info = read_video(video_path, pts_unit="sec")

    # Assume RGB video, normalize pixel values to [0, 1], and convert to torch tensor
    video = video.permute(0, 3, 1, 2) / 255.0
    video_tensor = torch.from_numpy(video).float()

    return video_tensor

# Load the InceptionI3d model
i3d_model = InceptionI3d(num_classes=400, final_endpoint="Mixed_5c", include_embds=True)

# Load pre-trained weights (replace 'i3d.pth.tar' with your actual checkpoint file)
checkpoint = torch.load('i3d.pth.tar')

i3d_model.load_state_dict(checkpoint['logits.conv3d.weight'])
i3d_model.eval()

# Path to the input video
video_path = 'input.mp4'  # Replace with the path to your video

# Load and preprocess the video
input_video = load_and_preprocess_video(video_path)

# Add batch dimension
input_video = input_video.unsqueeze(0)

# Extract features from InceptionI3d
with torch.no_grad():
    output = i3d_model(input_video)

# Print the features
print("Logits:", output["logits"])

# If include_embds is True, print embeddings as well
if i3d_model.include_embds:
    print("Embeddings:", output["embds"])


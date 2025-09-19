import torch
from utils import load_model, load_rgb_video, prepare_input, sliding_windows
from pathlib import Path
import json
import cv2

# Load JSON file
json_file_path = '/home/uday/Desktop/SignSpotting/bsldict/bsldict/word_index.json'  # Replace with the actual path to your JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Take input from the user
user_input = input("Enter the value to search: ")

# Search for the index in the JSON file
index = None
for item in data:
    if item["Word"] == user_input:
        index = item["Index"]
        break

# Save the result in one variable
result_variable = index if index is not None else "Word not found in the JSON file"

# Print or use the result as needed
print("Result:", result_variable)

# Constants
CHECKPOINT_PATH = Path('/home/uday/Desktop/SignSpotting/bsldict/models/i3d_mlp.pth.tar')
VIDEO_PATH = Path('/home/uday/Desktop/SignSpotting/bsldict/models/input.mp4')
ARCH = 'i3d_mlp'  # Architecture type
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_video(input_file, output_file, start_time, end_time):
    # Load the video clip
    video_clip = VideoFileClip(input_file)

    # Trim the video clip based on start and end times
    trimmed_clip = video_clip.subclip(start_time, end_time)

    # Write the trimmed video to a new file
    trimmed_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

    # Close the video clip
    video_clip.close()

# Example usage:
input_video_path = "/home/uday/Desktop/SignSpotting/bsldict/demo/sample_data/input.mp4"
output_video_path = "/home/uday/Desktop/SignSpotting/bsldict/demo/sample_data/ouuuu4.mp4"



def main():
    # Load the combined i3d_mlp model with pre-trained weights
    model = load_model(checkpoint_path=CHECKPOINT_PATH, arch=ARCH)

    # Load and preprocess the input video
    rgb_video = load_rgb_video(VIDEO_PATH, fps=25)
    rgb_input = prepare_input(rgb_video)

    # Apply sliding windows to the input video
    rgb_slides, t_mid = sliding_windows(rgb=rgb_input, stride=1, num_in_frames=16)

    # Store the timestamps for the first and last segments where the predicted class matches the index
    start_timestamp = None
    end_timestamp = None

    # Process each segment through the model and find the class with the highest probability
    for i, segment in enumerate(rgb_slides):
        segment = segment.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(segment)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
            
            # Find the class index with the highest probability for each segment
            highest_prob_class_indices = torch.argmax(probabilities, dim=1)
            
            if highest_prob_class_indices.item() == result_variable:
                # Calculate the timestamp in the original video
                timestamp_in_original_video = i / 25.0  # Assuming 25 frames per second
                print(f"Timestamp for segment {i}: {timestamp_in_original_video:.2f} seconds")

                # Update the start and end timestamps
                if start_timestamp is None:
                    start_timestamp = timestamp_in_original_video
                end_timestamp = timestamp_in_original_video

            # Optional: You can also check the value of the highest probability
            highest_prob_value = probabilities[0, highest_prob_class_indices.item()]
            #print("Highest Probability Value:", highest_prob_value.item())
    print(start_timestamp,end_timestamp)
    extract_video(input_video_path, output_video_path, start_timestamp, end_timestamp)

if __name__ == "__main__":
    main()


import math
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from PIL import Image
from tqdm import tqdm

sys.path.append("..")
from bsldict.download_videos import download_hosted_video, download_youtube_video
from models.i3d_mlp import i3d_mlp
from models.i3d import InceptionI3d

#checkpoint_path = Path("/home/uday/Desktop/SignSpotting/bsldict/models/i3d.pth.tar")
checkpoint_path = Path(r"C:\Users\91879\OneDrive\Documents\MY WHATTSPP FILES\sign-spot-main.zip\sign-spot-main\models/i3d.pth.tar")
checkpoint = torch.load(checkpoint_path)
print(checkpoint.keys())


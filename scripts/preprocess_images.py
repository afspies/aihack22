# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from skimage import filters
import cv2
from tqdm import tqdm

# %%
CLEANED_DIR = "trajectories_cleaned2"
os.mkdir(CLEANED_DIR)
# %%


def clean_frame(frame):
    filtered_array = median_filter(frame, size=10)
    kernel = np.ones((6, 6), np.float32)
    dst = cv2.filter2D(filtered_array, -1, kernel)
    res = cv2.resize(dst, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_uint8 = res.astype(bool).astype(np.uint8)
    return img_uint8


# %%
IMG_SIZE = 64
root_directory = "trajectories/"
trajectory_folders = os.listdir(root_directory)
for t_folder in tqdm(trajectory_folders):
    frames_in_trajectory = os.listdir(os.path.join(root_directory, t_folder))
    folder_name = os.path.join(CLEANED_DIR, t_folder)
    os.mkdir(folder_name)
    for i, frame_name in enumerate(frames_in_trajectory):
        filename = f"trajectory{t_folder[1:]}_frame{i:04}.npy"
        filepath = os.path.join(folder_name, filename)
        frame_file = np.load(os.path.join(root_directory, t_folder, frame_name))
        cleaned_frame = clean_frame(frame_file)
        np.save(filepath, cleaned_frame)


# %%
coords = []
for i in range(IMG_SIZE):
    for j in range(IMG_SIZE):
        coords.append(f"{i}-{j}")
idxs = list(range(IMG_SIZE**2))
coord2idx = dict(zip(coords, idxs))
coord2idx["63-63"]

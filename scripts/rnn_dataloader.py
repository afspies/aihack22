import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, train_test) -> None:
        super().__init__()
        self.timeframes_to_get = 100
        IMG_SIZE = 64
        self.VECTOR_SIZE = IMG_SIZE ** 2
        coords = []
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                coords.append(f"{i}-{j}")
        idxs = list(range(IMG_SIZE**2))
        self.coord2idx = dict(zip(coords, idxs))

        if train_test == "train":
            trajectories_to_get = list(range(1, 16)) + list(range(24, 40))
        else:
            trajectories_to_get = list(range(16, 24)) + list(range(40, 49))

        self.filenames_list = []
        root_directory = "trajectories_cleaned2/"
        trajectory_folders = os.listdir(root_directory)
        for t_folder in tqdm(trajectory_folders):
            if not (int(t_folder[1:]) in trajectories_to_get):
                continue
            frames_in_trajectory = os.listdir(os.path.join(root_directory, t_folder))
            full_path = os.path.join(root_directory, t_folder)
            filepaths = [os.path.join(full_path, fname) for fname in frames_in_trajectory]
            self.filenames_list.append(filepaths)

    def pad_and_khot(self, coord_list_for_sample):
        # coord_list_for_sample = [[0, 40], [0, 41], [0, 42], [50, 40], [50, 41], [50, 42]]
        # todo: pad, and then khot
        zeros = torch.zeros(self.VECTOR_SIZE)
        for coord in coord_list_for_sample:
            composed_string = f"{coord[0]}-{coord[1]}"
            idx = self.coord2idx[composed_string]
            zeros[idx] = 1
        return zeros

    def __len__(self):
        len_ = 0
        for list_of_frames in self.filenames_list:
            len_list_of_frames = len(list_of_frames)
            len_ += len_list_of_frames
        return len_

    def get_relevant_sublist(self, index):
        total_items = 0
        for i, trajectory in enumerate(self.filenames_list):
            len_trajectory = len(trajectory)
            total_items += len_trajectory
            if total_items > index:
                relevant_sublist = self.filenames_list[i - 1]
                to_index_from_relevant_sublist = index - (total_items - len_trajectory)
                break
        return relevant_sublist, to_index_from_relevant_sublist

    def __getitem__(self, index):
        relevant_sublist, to_index_from_relevant_sublist = self.get_relevant_sublist(index)
        sample_items = relevant_sublist[to_index_from_relevant_sublist:to_index_from_relevant_sublist + self.timeframes_to_get]
        # check if len(sample_items) < self.timeframes_to_get
        # If so, we've got one of the last frames from the video
        # We will then just take the preceeding frames before the index to create len(sample_items) == self.timeframes_to_get
        if len(sample_items) < self.timeframes_to_get:
            diff = self.timeframes_to_get - len(sample_items)
            additional_sample_items = sample_items[index - diff:index]
            sample_items = additional_sample_items + sample_items

        full_sample = torch.zeros(self.timeframes_to_get, self.VECTOR_SIZE)
        for i, item in enumerate(sample_items):
            img = np.load(item)
            coords = np.transpose(img.nonzero())
            khot_items = self.pad_and_khot(coords)
            full_sample[i] = khot_items

        return full_sample

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, Dataloader
from torchvision import transforms, utils


def create_csv(root_dir):
    '''
    Creates the train/val/test csvs
    '''


class Bubbles(Dataset):
    """Bubble dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with file names to load
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_files = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = os.path.join(self.root_dir,
                                self.data_files.iloc[idx, 0])
        # load video
        video_instance = np.load(file_path)
        print(video_instance)
        print(np.shape(video_instance))
        if self.transform:
            sample = self.transform(sample)

        return sample



if __name__ == '__main__':
    pass
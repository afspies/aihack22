import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def visualize_rollout(rollout, interval=50, show_step=False):
    """Visualization for a single sample rollout of a physical system.
 
    Args:
        rollout (numpy.ndarray): Numpy array containing the sequence of images. It's shape must be
            (seq_len, height, width, channels).
        interval (int): Delay between frames (in millisec).
        show_step (bool): Whether to draw the step number in the image
    """
    fig = plt.figure()
    img = []
    for i, im in enumerate(rollout):
        if show_step:
            black_img = np.zeros(list(im.shape))
            cv2.putText(
                black_img, str(i), (0, 30), fontScale=0.22, color=(255, 255, 255), thickness=1,
                fontFace=cv2.LINE_AA)
            res_img = (im + black_img / 255.) / 2
        else:
            res_img = im
        img.append([plt.imshow(res_img, animated=True)])
    ani = animation.ArtistAnimation(fig,
                                    img,
                                    interval=interval,
                                    blit=True,
                                    repeat_delay=100)
    plt.show()


def plot_frame(frame, i):

    fig, ax = plt.subplots(1,1)
    plt.imshow(frame)
    plt.savefig(f'figs/{i}')
    plt.close()

def collate_trajectories(root_dir, new_dir):
    '''
    structure of files is:
    trajectories-
        traj1-
            frame1-
            frame2-
            frame3-
        traj2-
            frame1
            frame2
    This func creates a new folder of the structure
    
    trajectories_collate-
        traj1.npy
        traf2.npy
        traj3.npy
    '''
    if not os.path.exists(new_dir):
        print('creating new directory for collated frames')
        os.makedirs(new_dir)
        

    for directory in os.listdir(root_dir):
        concat_list = []
        for file in os.listdir(os.path.join(root_dir, directory)): 
            frame = np.load(os.path.join(root_dir, directory, file))
            concat_list.append(frame)
                
        concat_traj = np.stack(concat_list, axis=0)
        print(np.shape(concat_traj))
        #visualize_rollout(np.expand_dims(concat_traj, axis=-1))
        np.save(os.path.join(new_dir, directory), concat_traj)
                



class Bubbles(Dataset):
    """Bubble dataset."""

    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            csv_file (string): Path to the csv file with file names to load
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if not os.path.exists(root_dir):
            print('you first need to collate the npy files. please run collate_trajectories')
        self.split = split
        self.root_dir = root_dir
        self.data = self.create_train_test()

        print(self.data)
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.data[idx]
        traj = np.load(file_path)
        if self.transform:
            sample = self.transform(sample)

        return sample

    def create_train_test(self):
        '''
        Creates the train/test lists
        16 - 23 test 
        40 -48  test
        the rest is train
        '''
        test1 = list(range(16,24))
        test2 = list(range(40,49))
        test = test1 + test2

        train = list(range(1,16)) + list(range(24,40))
        test = [os.path.join(self.root_dir, f't{i}.npy') for i in test]
        
        train = [os.path.join(self.root_dir, f't{i}.npy') for i in train]
        if self.split == 'train':
            return train
        else:
            return test


if __name__ == '__main__':
    e = Bubbles('/vol/bitbucket/hgc19/aihack22/data/trajectories_full')
    #collate_trajectories('/vol/bitbucket/hgc19/aihack22/data/trajectories', '/vol/bitbucket/hgc19/aihack22/data/trajectories_full' )
    # create_train_test('/vol/bitbucket/hgc19/aihack22/data/trajectories_full')

    data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        #now do transforms
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.Pad(4,0),
        transforms.ToTensor(),
    ]),
    }

    train_loader = Dataloader(e, batch=1, shuffle=True)
    for i_batch, data in enumerate(train_loader):
        print(i_batch, data.size())
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
import random

from tqdm import tqdm
import cv2

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
    return fig


def plot_frame(frame, i):

    fig, ax = plt.subplots(1,1)
    plt.imshow(frame)
    plt.savefig(f'figs/{i}')
    plt.close()

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tesnor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True
                
class RandomRotationVideo(object):
    """
    Rotate a video about the H,W dim by 90 degrees k times
    """

    def __init__(self):
        pass
    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        k =  random.sample([0,1,2,3], 1)
        assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
        clip = torch.rot90(clip, k[0], [-2,-1])
        return clip

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)

class RandomVerticalFlipVideo(object):
    """
    Flip the video clip along the Vertical direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
  
        if random.random() < self.p:
            assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
            clip = torch.flip(clip, [-2])
        return clip

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)

class RandomHorizontalFlipVideo(object):
    """
    Flip the video clip along the horizonal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        if random.random() < self.p:
            assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
            clip = torch.flip(clip, [-1])
        return clip

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)




class Bubbles(Dataset):
    """Bubble dataset."""

    def __init__(self, frame_dir, root_dir, transform=None, split='train', max_len=948):
        """
        Args:
            csv_file (string): Path to the csv file with file names to load
            frame_dir (string): Path to directory with frames per npj
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.max_len = max_len
        if not os.path.exists(root_dir):
            print('you first need to collate the npy files. please run collate_trajectories')
            self.collate_trajectories(frame_dir, root_dir)
        self.split = split
        self.root_dir = root_dir
        self.data = self.create_train_test()

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.data[idx]
        traj = np.load(file_path)


        traj = torch.from_numpy(traj)
        traj = torch.unsqueeze(traj, 1)
        if self.transform != None:
            traj = self.transform(traj)

        return traj

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

    def collate_trajectories(self, root_dir, new_dir):
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
            
        max_len = 0
        for directory in tqdm(os.listdir(root_dir)):
            concat_list = []
            for file in sorted(os.listdir(os.path.join(root_dir, directory))):
                print(file)
                frame = np.load(os.path.join(root_dir, directory, file))
                frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
                concat_list.append(frame)
                    
            concat_traj = np.stack(concat_list, axis=0)
            max_len = max(max_len, np.shape(concat_traj)[0])
            #visualize_rollout(np.expand_dims(concat_traj, axis=-1))
            concat_traj = self.pad_traj(concat_traj)
            np.save(os.path.join(new_dir, directory), concat_traj)
    
    def pad_traj(self, traj):
        temp_traj = np.zeros((self.max_len, np.shape(traj)[-1], np.shape(traj)[-1]))
        temp_traj[:np.shape(traj)[0], :, :] = traj
        return temp_traj

if __name__ == '__main__':
    # e = Bubbles('/vol/bitbucket/hgc19/aihack22/data/trajectories', '/vol/bitbucket/hgc19/aihack22/data/trajectories_full_128by128_2')
    # #collate_trajectories('/vol/bitbucket/hgc19/aihack22/data/trajectories', '/vol/bitbucket/hgc19/aihack22/data/trajectories_full' )
    # # create_train_test('/vol/bitbucket/hgc19/aihack22/data/trajectories_full')


    # train_loader = DataLoader(e, batch_size=4, shuffle=True)
    # for i_batch, data in enumerate(train_loader):
    #     print(i_batch, data.size())

    x = torch.randn((1,2,3,3))
    t = RandomRotationVideo() 
    print(x)
    print(t(x))
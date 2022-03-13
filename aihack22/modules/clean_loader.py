import torch
import numpy as np
import os
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset



def main():
    cfg = {'batch_size': 10}
    dl = SequenceChopDataloader('/media/home/alex/aihack22/data/', cfg).get_dataloader()
    dl = iter(dl)
    # print(next(dl))

splits = {'train': list(range(0,16)) + list(range(24,40)),
          'test': list(range(16,24)) + list(range(40,48))}
labels = {'train': {k:int(k>=24) for k in (list(range(0,16)) + list(range(24,40)))},
          'test' : {k:int(k>=40) for k in (list(range(16,24)) + list(range(40,48)))}
        }
time_skip = 5
resolution = (64,64)


class SequenceChopDataloader():
    def __init__(self, base_data_dir, cfg, split='train') -> None:
        self.split = split
        self.max_traj = 948
        self._load_dataset(base_data_dir, split)
        self._instantiate_dataloader(cfg, split)
    
    def get_dataloader(self):
        return self.dataloader

    def get_dataset(self):
        return self.dataset

    def _instantiate_dataloader(self, cfg, split):
        self.dataloader =  DataLoader(self.dataset, batch_size=cfg['batch_size'], shuffle=(split=='train'))
    
    def _load_dataset(self, base_data_dir, split):
        data = np.load(Path(base_data_dir)/'cleaned_64x64.npy', allow_pickle=True)
        data = [data[i] for i in splits[split]]
        short_trajs = []
        traj_labels = []
        for i, traj in enumerate(data): #! Assuming ordered
            traj_id = splits[split][i]
            seq_len = len(traj)
            mod = seq_len%time_skip
            # traj = traj[:seq_len-mod, :,:]
            traj = traj[:100+time_skip, :,:]
            traj = traj.reshape(len(traj)//time_skip,time_skip,*resolution).transpose(1,0,2,3)[:,:,None,:,:]
            short_trajs.append(torch.tensor(traj))
            traj_labels.append(torch.tile(torch.tensor(labels[split][traj_id]), (traj.shape[0],)))

        trajectories = torch.concat(short_trajs, dim=0)
        traj_labels = torch.concat(traj_labels, dim=0)
        self.dataset = TensorDataset(trajectories, traj_labels)

    # Handle Repeat etc. by checking/catching raise StopIteration?
    def __next__(self):
        try:
            batch = next(self.ds_iter)
        except StopIteration:
            self.ds_iter = iter(self.dataloader)
            raise StopIteration
        return batch
   
    # def _create_train_test(self):
    #     '''
    #     Creates the train/test lists
    #     16 - 23 test 
    #     40 -48  test
    #     the rest is train
    #     '''
    #     test1 = list(range(16,24))
    #     test2 = list(range(40,49))
    #     test = test1 + test2

    #     train = list(range(1,16)) + list(range(24,40))
    #     test = [os.path.join(self.root_dir, f't{i}.npy') for i in test]
        
    #     train = [os.path.join(self.root_dir, f't{i}.npy') for i in train]
    #     if self.split == 'train':
    #         return train
    #     else:
    #         return test

if __name__ == '__main__':
    main()

from pathlib import Path
import os
import numpy as np
import random
import pickle
import subprocess
import os
import wandb
import torch

# -- Misc. -- # 
def set_rng_seeds(seed=42):
  random.seed(seed)
  np.random.seed(seed)
#   tf.random.set_seed(seed)
#   tf.experimental.numpy.random.seed(seed)
  
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'


# This function should be called after all imports,
# in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2):
    """Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable

    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
                                              
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
    """
    # Get the list of GPUs via nvidia-smi
    smi_query_result = subprocess.check_output('nvidia-smi -q -d Memory | grep -A4 GPU', shell=True)
    # Extract the usage information
    gpu_info = smi_query_result.decode('utf-8').split('\n')
    gpu_info = list(filter(lambda info: 'Used' in info, gpu_info))
    gpu_info = [int(x.split(':')[1].replace('MiB', '').strip()) for x in gpu_info] # Remove garbage
    gpu_info = gpu_info[:min(max_gpus, len(gpu_info))] # Limit to max_gpus
    # Assign free gpus to the current process
    gpus_to_use = ','.join([str(i) for i, x in enumerate(gpu_info) if x < threshold_vram_usage])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
    print(f'Using GPUs {gpus_to_use}' if gpus_to_use else 'No free GPUs found')
    return gpus_to_use

def save_model(model, path):
  if not os.path.exists(f'./data/{path}'):
    os.makedirs(f'./data/{path}')

  model_path = f'./data/{path}/model.pth'
  torch.save(model.state_dict(), model_path)
  artifact = wandb.Artifact(path, type='model')
  artifact.add_file(model_path)
  return artifact
from pathlib import Path
from typing import Type
import wandb
from datetime import datetime
import os
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import aihack22 as ah # pip install -e ./

# Options
DEBUG = False
config = dict( # Configurations will be taken from config yaml files in future.
    epochs=5,
    classes=10,
    kernels=[16, 32],
    batch_size=128,
    learning_rate=0.005,
    rng_seed=42,
    dataset="MNIST",
    architecture="CNN")
wandb_project = 'conv_setup'
wandb_run_name = f"exp1-{datetime.now().strftime('%H_%M')}"
if DEBUG:
    os.environ['WANDB_MODE']='disabled'
os.environ["WANDB_RUN_GROUP"] = config['architecture']

data_path = os.path.join(Path(ah.__path__[0]).parent/'data') #! Don't git push data please.
model_make_fn = ah.networks.make_cnn_model # fn should return a tuple of (model, criterion, optimizer)

# Initialize PRNG seeds and select free GPUs on machine
ah.set_rng_seeds(config['rng_seed'])
free_gpus = ah.assign_free_gpus()
device = 'cpu' if free_gpus == '' else 'cuda'

def main():
    if not os.path.exists('./data'):
        os.makedirs('./data')
    # Build, train and analyze the model with the pipeline
    model = model_pipeline(config, model_make_fn, track_gradients=False)


def model_pipeline(hyperparameters, model_fn, track_gradients=False):
    if not os.makedirs('./data/{wandb_run_name}'):
        os.makedirs(f'./data/{wandb_run_name}')

    # tell wandb to get started
    with wandb.init(project=wandb_project, dir=f'./data/{wandb_run_name}', name=wandb_run_name, entity='aihack', config=hyperparameters) as run:
        config = wandb.config

        train_loader, test_loader = load_ds(config) 
        model, criterion, optimizer = model_fn(config, device)
        # print(model)
        if track_gradients:
            wandb.watch(model, log="all", log_freq=1000)

        train(model, train_loader, criterion, optimizer, config)
        test(model, test_loader)
        if not DEBUG:
            run.log_artifact(ah.save_model(model, wandb_run_name))

    return model

# ! this and fetch data need to be updated to match our dataloader
def load_ds(config):
    # Make the data
    train, test = fetch_data(config, train=True), fetch_data(config, train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    return train_loader, test_loader 

def fetch_data(config, slice=5, train=True):
    if config['dataset']=='MNIST':
        #! temp for testing - remove slow mirror from list of MNIST mirrors
        torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                            if not mirror.startswith("http://yann.lecun.com")]
        full_dataset = torchvision.datasets.MNIST(root="./data",
                                                train=train, 
                                                transform=transforms.ToTensor(),
                                                download=True)
        #  equiv to slicing with [::slice] 
        sub_dataset = torch.utils.data.Subset(
        full_dataset, indices=range(0, len(full_dataset), slice))
        
        return sub_dataset
    else:
        raise ValueError('HARRY')


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader

def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    # total_batches = len(loader) * config.epochs
    steps = 0
    for epoch in range(config.epochs):   
        with tqdm(loader, unit="batch") as tepoch: 
            for batch_idx, (images, labels) in enumerate(tepoch):
                loss = train_batch(images, labels, model, optimizer, criterion)

                steps += 1                
                if ((batch_idx + 1) % 25) == 0: # Report metrics every 25th batch
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(loss=loss.item()/len(images))
                    wandb.log({"epoch": epoch, "loss": loss}, step=steps)
                    # print(f"Loss after {str(example_ct).zfill(5)}" + f" examples: {loss:.3f}")

def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass 
    optimizer.zero_grad()
    loss.backward()
    # Step with optimizer
    optimizer.step()
    return loss

def test(model, test_loader):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {100 * correct / total}%")
        
        wandb.log({"test_accuracy": correct / total})

if __name__ == "__main__":
    main()
from random import random
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from aihack22.networks import EncoderRNN, repackage_hidden
from rnn_dataloader import MyDataset
import torch.nn as nn
import torchvision


class RNNTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderRNN()
        self.hidden = repackage_hidden(self.encoder.init_hidden())
        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.MSELoss()
        self.teacher_forcing_ratio = 0.5

    def training_step(self, batch, batch_idx):
        # example batch:
        # [[[0 0 1 1 1 0 0... 0 1 0], [0 0 1 1 1 0 0... 0 1 0], [0 0 1 1 1 0 0... 0 1 0]]] # [B, T, 4096]
        src = batch[:, :-1]
        trg = batch[:, 1:]

        total_loss = 0
        encoder_outputs = src[:, 0].unsqueeze(1)
        for t in range(src.shape[1]):
            if random() > self.teacher_forcing_ratio:
                encoder_outputs, self.hidden = self.encoder(src[:, t], self.hidden)
            else:
                encoder_outputs, self.hidden = self.encoder(encoder_outputs, self.hidden)
            diff = (trg[:, t] - src[:,t]).astype(bool).astype()
            loss = self.criterion(encoder_outputs, trg[:, t].unsqueeze(1))
            total_loss += loss

        self.hidden = repackage_hidden(self.encoder.init_hidden())
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        src = batch[:, :-1]
        trg = batch[:, 1:]

        total_loss = 0

        encoder_outputs = src[:, 0].unsqueeze(1)
        all_outputs = [encoder_outputs]
        for t in range(src.shape[1]):
            encoder_outputs, self.hidden = self.encoder(encoder_outputs, self.hidden)
            loss = self.criterion(encoder_outputs, trg[:, t].unsqueeze(1))
            total_loss += loss
            all_outputs.append(encoder_outputs)
        self.log("val_loss", total_loss)
        return all_outputs

    def validation_epoch_end(self, outputs):
        sample_imgs = outputs[:6]
        collected_images = []
        for image in sample_imgs:
            image = torch.stack(image).permute(1, 0, 2, 3)
            bs, t, c, hw = image.shape[0], image.shape[1], image.shape[2], image.shape[3]
            image = image.reshape(bs, t, c, 64, 64)
            collected_images.append(image)

        collected_images = torch.cat(collected_images, 0)
        self.logger.experiment.add_video('generated_image', collected_images, self.global_step)
        print()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    train_dataset = MyDataset("train")
    val_dataset = MyDataset("val")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    rnn = RNNTrainer()
    trainer = pl.Trainer(val_check_interval=1000, gpus=1, limit_val_batches=100)
    trainer.fit(rnn, train_dataloader, val_dataloader)
    print()

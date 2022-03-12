import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from aihack22.networks import EncoderRNN, repackage_hidden
from rnn_dataloader import MyDataset
import torch.nn as nn


class RNNTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderRNN()
        self.hidden = repackage_hidden(self.encoder.init_hidden())
        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")

    def training_step(self, batch, batch_idx):
        # example batch:
        # [[[0 0 1 1 1 0 0... 0 1 0], [0 0 1 1 1 0 0... 0 1 0], [0 0 1 1 1 0 0... 0 1 0]]] # [B, T, 4096]
        src = batch[:, :-1]
        trg = batch[:, 1:]

        total_loss = 0
        for t in range(src.shape[1]):
            encoder_outputs, self.hidden = self.encoder(src[:, t], self.hidden)
            loss = self.criterion(encoder_outputs, trg[:, t].unsqueeze(1))
            total_loss += loss

        self.hidden = repackage_hidden(self.encoder.init_hidden())
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        src = batch[:, :-1]
        trg = batch[:, 1:]

        total_loss = 0
        for t in range(src.shape[1]):
            encoder_outputs, self.hidden = self.encoder(src[:, t], self.hidden)
            loss = self.criterion(encoder_outputs, trg[:, t].unsqueeze(1))
            total_loss += loss

        self.log("val_loss", total_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    train_dataset = MyDataset("train")
    val_dataset = MyDataset("val")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    rnn = RNNTrainer()
    trainer = pl.Trainer(val_check_interval=1000)
    trainer.fit(rnn, train_dataloader, val_dataloader)
    print()

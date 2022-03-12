from turtle import forward
import torch
import torch.nn as nn


class MyRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4096, 100)
        self.rnn = nn.RNN(100, 100, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(100, 4096)

    def forward(self, x, hidden):
        # shape(x) = [B x T x 4096]
        embedded = self.dropout(self.embedding(x))
        # shape(embedded) = [B x T x 100]
        output, hidden = self.rnn(embedded, hidden)
        # shape(output) = [B x T x output]
        pred = self.fc(output)

        return pred, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, 1, 100)

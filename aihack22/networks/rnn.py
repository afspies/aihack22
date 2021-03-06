import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    def __init__(self, input_size=4096, hidden_size=300):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, num_layers=3)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, input_size),

        )

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        output = self.relu(output)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(3, 1, self.hidden_size).to("cuda")


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

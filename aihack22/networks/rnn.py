import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    def __init__(self, input_size=4096, hidden_size=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, input_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

from torch.utils.data import Dataset, DataLoader
from Utils import shuffle_for_training, vectorize_dataset
import torch


class CharDataset(Dataset):
    def __init__(self, X, Y, max_len):
        self.max_len = max_len
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

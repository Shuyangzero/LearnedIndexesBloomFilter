import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn.functional as func
from DataLoader import TestDataset
from tqdm import tqdm

class GRUModel(nn.Module):
    def __init__(self, embeddings_path, embedding_dim, char_indices, indices_char, maxlen=60, hidden_size=16):
        super(GRUModel, self).__init__()
        self.embeddings_path = embeddings_path
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.model = None
        self.hidden_size = hidden_size
        self.char_indices = char_indices
        self.indices_char = indices_char
        self.num_chars = len(self.char_indices)
        self.embedding_vectors = {}
        with open(self.embeddings_path, 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                vec = np.array(line_split[1:], dtype=float)
                char = line_split[0]
                self.embedding_vectors[char] = vec

        embedding_matrix = np.zeros((self.num_chars + 1, self.embedding_dim))
        for char, i in self.char_indices.items():
            embedding_vector = self.embedding_vectors.get(char)
            assert(embedding_vector is not None)
            embedding_matrix[i] = embedding_vector

        self.embeddings = nn.Embedding(self.num_chars + 1, self.embedding_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embeddings.weight.requires_grad = False
        self.gru = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 2)
        self.fc4 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.embeddings(x)
        x = x.permute(1, 0, 2)
        _, last_hidden = self.gru(x)
        x = last_hidden.permute(1, 2, 0).squeeze(-1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.fc2(x)
        x = func.relu(x)
        x = self.fc3(x)
        x = func.relu(x)
        x = self.fc4(x)
        x = func.sigmoid(x)
        return x

    def predict(self, text_x):
        x = np.zeros((1, self.maxlen), dtype=np.int)
        offset = max(self.maxlen - len(text_x), 0)
        for t, char in enumerate(text_x):
            if t >= self.maxlen:
                break
            x[0, t + offset] = self.char_indices[char]
        pred = self.model(x)
        return pred[0][0]

    def predicts(self, text_X, device, batch_size=1024):
        X = np.zeros((len(text_X), self.maxlen), dtype=np.int)
        for i in range(len(text_X)):
            offset = max(self.maxlen - len(text_X[i]), 0)
            for t, char in enumerate(text_X[i]):
                if t >= self.maxlen:
                    break
                X[i, t + offset] = self.char_indices[char]
        
        dev_data = TestDataset(X)
        dataloader = DataLoader(
            dev_data, batch_size=batch_size, shuffle=True, num_workers=0)
        preds = []
        for x in tqdm(dataloader):
            x.to(device)
            out = self.forward(x)
            out.detach().numpy()
            preds.extend(list(out.detach().numpy().reshape((-1,))))
        return preds

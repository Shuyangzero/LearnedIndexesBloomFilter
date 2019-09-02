from Utils import *
from DataLoader import CharDataset
from torch.utils.data import DataLoader
import json
from GRUModel import GRUModel
from tqdm import tqdm
from tqdm import trange
import torch
from torch import nn


def train(model, train_dataloader, lr=0.001):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCELoss()
    for i in range(epochs):
        running_loss = 0
        for data in tqdm(train_dataloader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            running_loss = loss.float()
            optimizer.step()
        print("loss is {}".format(running_loss/float(len(train_dataloader))))
    torch.save(model, "model.h5")


if __name__ == "__main__":
    train_mode = False
    max_len = 60
    embedding_dim = 50
    batch_size = 1024
    epochs = 30
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('./data/dataset.json', 'r') as f:
        dataset = json.load(f)
    positives = dataset['positives']
    negatives = dataset['negatives']
    negatives_train = negatives[0: int(len(negatives) * .9)]
    negatives_dev = negatives[int(
        len(negatives) * .8): int(len(negatives) * .9)]
    negatives_test = negatives[int(len(negatives) * .9):]

    if train_mode:
        shuffle = shuffle_for_training(negatives_train, positives)
        X, Y, char_indices, indices_char = vectorize_dataset(
            shuffle[0], shuffle[1], max_len)
        train_data = CharDataset(X, Y, max_len)
        train_dataloader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        model = GRUModel(
            './data/glove.6B.50d-char.txt', embedding_dim, char_indices, indices_char)
        model.to(device)
        train(model, train_dataloader)
    else:
        model_path = "./model.h5"
        model = torch.load(model_path, map_location=device)
    #model.predicts(negatives_dev, device)
    evaluate_model(model, positives, negatives_train,negatives_dev, negatives_test, device, threshold = 0.5)

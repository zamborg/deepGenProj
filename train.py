import os
import sys
import numpy as np
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, concatenate_datasets
import numpy as np
from dotenv import load_dotenv
import wandb
from dataclasses import dataclass

BERT = "distilbert-base-uncased"
GEMMA2B = "google/gemma-2b" # requires hftoken
HFTOKEN = os.environ['HF_TOKEN'] if 'HF_TOKEN' in os.environ.keys() else None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class detector(nn.Module):
    def __init__(self, hdim, odim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hdim, hdim),
            torch.nn.BatchNorm1d(hdim),
            torch.nn.Dropout(0.2),
            torch.nn.GELU(),
            
            # torch.nn.Linear(hdim, 2*hdim),
            # torch.nn.BatchNorm1d(2*hdim),
            # torch.nn.GELU(),

            # torch.nn.Linear(2*hdim, 4*hdim),
            # torch.nn.BatchNorm1d(4*hdim),
            # torch.nn.Dropout(0.2),
            # torch.nn.GELU(),

            # torch.nn.Linear(4*hdim, 4*hdim),
            # torch.nn.BatchNorm1d(4*hdim),
            # torch.nn.Dropout(0.2),
            # torch.nn.GELU(),

            # torch.nn.Linear(4*hdim, 4*hdim),
            # torch.nn.BatchNorm1d(4*hdim),
            # torch.nn.Dropout(0.2),
            # torch.nn.GELU(),

            # torch.nn.Linear(4*hdim, 2*hdim),
            # torch.nn.BatchNorm1d(2*hdim),
            # torch.nn.GELU(),

            # torch.nn.Linear(2*hdim, hdim),
            # torch.nn.BatchNorm1d(hdim),
            # torch.nn.GELU(),

            torch.nn.Linear(hdim, 100),
            torch.nn.GELU(),
            torch.nn.Linear(100, odim)
        )
    
    def forward(self, X):
        return self.net(X)

class embedder():
    def __init__(self, module_name, device):
        self.model = AutoModel.from_pretrained(module_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(module_name)
        self.device = device
        self.mean = True 
    
    def forward(self, X):
        tokens = self.tokenizer(X, return_tensors='pt', padding=True, truncation=True, max_length=256).to(self.device)
        if self.mean:
            return self.model(**tokens).last_hidden_state.mean(dim=1)
        return self.model(**tokens).last_hidden_state
    

class CivilComments():
    def __init__(self) -> None:
        dataset = load_dataset('civil_comments')
        self.dataset = concatenate_datasets([dataset['train'], dataset['test']])
    @staticmethod
    def get_y(datum):
        return torch.tensor(np.nan_to_num(np.array(list(datum.values())[1:])), dtype=torch.float).T
    def get_loader(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size)


@dataclass
class Config:
    model_name: str
    batch_size: int
    epochs: int
    lr: float
    criterion: str
    eta: bool = False

def train(model_name, batch_size, epochs, lr, criterion, eta):
    wandb.init(project="deepGen", tags=[criterion])
    start_time = time.time()
    embed = embedder(model_name, DEVICE)
    print(f"Embedder Loaded in {time.time() - start_time} seconds")
    
    start_time = time.time()
    dataset = CivilComments()
    loader = dataset.get_loader(batch_size)
    print(f"Dataset Loaded in {time.time() - start_time} seconds")

    start_time = time.time()
    if model_name == BERT:
        hdim = 768
        model = detector(hdim, 7).to(DEVICE)
    elif model_name == GEMMA2B:
        hdim = 2048
        model = detector(hdim, 7).to(DEVICE)
    wandb.watch(model, log="all")
    
    criterion = nn.MSELoss(reduction='sum') if criterion == 'mse' else nn.Softmax(dim=1, reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Model Loaded in {time.time() - start_time} seconds")
    
    start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for i, batch in enumerate(loader):
            if eta and i == 10:
                tot = time.time() - start
                print(f"Epoch_ETA: {tot * (len(loader) - i) / 10} | ETA = {epochs * tot * (len(loader) - i) / 10}")
            X = embed.forward(batch['text'])
            y = dataset.get_y(batch).to(DEVICE)
            optimizer.zero_grad()
            yhat = model(X)
            loss = criterion(yhat, y)
            wandb.log({"batch_loss" : loss.item()})
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        wandb.log({"epoch_loss" : epoch_loss})


if __name__ == '__main__':
    config = Config(
        model_name=BERT,
        batch_size=32,
        epochs=2,
        lr=0.01,
        criterion='mse',
        eta=True
    )
    train(**config.__dict__)

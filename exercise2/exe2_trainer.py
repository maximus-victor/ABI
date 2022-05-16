import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Tuple, List
from tqdm.auto import tqdm
import json
import h5py
from exe2_network import Model
from pathlib import Path
from torch.nn import BCELoss


# Define a dataset tailored to the data that should be used
class FancyDataset(Dataset):
    # Dataset ... map-style dataset
    def __init__(self, h5_path: Path, json_path: Path):
        self.data = h5py.File(h5_path, 'r')
        # use as "index map"
        self.ids_list = list(self.data.keys())
        self.json_dict = FancyDataset.load_json(json_path)


    # return the number of elements in the dataset
    def __len__(self):
        return len(self.ids_list)

    # return item at specific index
    def __getitem__(self, idx: int):
        identifier = self.ids_list[idx]
        idx_element = torch.from_numpy(self.data[identifier][:])
        idx_json = self.json_dict[identifier]
        return idx_element, idx_json

    @staticmethod
    def load_json(json_path):
        with open(json_path, 'r') as file:
            return json.load(file)


#@title Trainer
class Trainer:
    def __init__(self, model: Any, device: torch.device, save_path: str = 'model.pth'):
        super(Trainer, self).__init__()

        self.device = device
        self.model = model.to(self.device)
        self.save_path = save_path

        self.sigmoid = nn.Sigmoid()
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, dataloader, num_epoch: int):
        self.model.train()
        for epoch in tqdm(range(num_epoch), desc='Train Epoch', position=0, leave=True, ascii=True):
            running_loss, num_seqs, correct = 0.0, 0, 0
            for i, data in enumerate(tqdm(dataloader, desc='Epoch Progress', position=0, leave=True, ascii=True), 0):
                
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.view(-1, 1).to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.bce_with_logits(outputs.float(), labels.float())
                loss.backward()
                self.optimizer.step()
                num_seqs += len(labels)

                running_loss += loss.item()
                outputs = (nn.Sigmoid()(outputs)>=0.5).float()
                correct += (outputs == labels).float().sum()

            print(f'[{epoch + 1}] loss: {running_loss / num_seqs:.5f}\tacc: {correct/num_seqs}')

        print('Finished Training')
        self.save_model()


    def save_model(self):
        if self.save_path:
            torch.save(self.model.state_dict(), self.save_path)


def test_model():
    print("Testing: ")
    with h5py.File('tests/data/val_dataloader.h5') as hf:
        val_dataloader = [(torch.from_numpy(data['emb'][:]), torch.from_numpy(data['lbl'][:]))
                for data_idx, data in hf['first'].items()]


    model = Model()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    running_loss, num_seqs, correct = 0.0, 0, 0

    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            inputs, labels = data

            outputs = model.predict(inputs)
            loss = BCELoss()(outputs.float(), labels.float())

            num_seqs += len(labels)
            running_loss += loss.item()


            outputs = (outputs>0.5).float()
            correct += (outputs == labels).float().sum()

    print("Loss", running_loss / num_seqs)
    print("Accuracy", (correct / num_seqs).item())




dataset = FancyDataset(Path("data/embeddings.h5"), Path("data/train_lbl.json"))
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

net = Model()

trainer = Trainer(net, torch.device('cpu'))
trainer.train(train_dataloader, 10)

test_model()
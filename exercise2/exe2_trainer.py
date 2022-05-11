import torch
from torch import nn
import torch.optim as optim
from typing import Any
from tqdm.auto import tqdm


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
            running_loss, num_seqs = 0.0, 0
            for i, data in enumerate(tqdm(dataloader, desc='Epoch Progress', position=0, leave=True, ascii=True), 0):
                inputs, labels = data
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.bce_with_logits(outputs, labels)
                loss.backward()
                self.optimizer.step()
                num_seqs += len(labels)

                running_loss += loss.item()
            print(f'[{epoch + 1}] loss: {running_loss / num_seqs:.3f}')

        print('Finished Training')
        self.save_model()

    def validate(self, val_dataloader):
        self.model.eval()
        running_loss, num_seqs = 0.0, 0
        for i, data in enumerate(val_dataloader):
            inputs, labels = data
            labels = labels.to(self.device)

            outputs = self.model(inputs)

            loss = self.bce_with_logits(outputs, labels)
            num_seqs += len(labels)

            running_loss += loss.item()

        return running_loss / num_seqs


    def save_model(self):
        if self.save_path:
            torch.save(self.model.state_dict(), self.save_path)

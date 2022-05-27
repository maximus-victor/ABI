import torch
from torch import nn
import torch.optim as optim
from typing import Any
from tqdm.auto import tqdm
import sklearn.metrics as met

class Trainer:
    def __init__(self, model: Any, device: torch.device, save_path: str = 'model.pth'):
        super(Trainer, self).__init__()

        self.device = device
        self.model = model.to(self.device)
        self.save_path = save_path

        self.sigmoid = nn.Sigmoid()
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, dataloader, num_epoch: int):
        self.model.train()
        for epoch in tqdm(range(num_epoch), desc='Train Epoch', position=0, leave=True, ascii=True):
            running_loss, num_seqs = 0.0, 0
            for i, data in enumerate(tqdm(dataloader, desc='Epoch Progress', position=0, leave=True, ascii=True), 0):
                inputs, labels = data
                labels = labels.view(-1, 1).to(self.device)

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
        self.optimizer.zero_grad()
        self.model.eval()
        running_loss, num_seqs = 0.0, 0
        prediction = None
        allLabels = None
        for i, data in enumerate(tqdm(val_dataloader, desc='Validation Progress', position=0, leave=True, ascii=True), 0):
            inputs, labels = data
            labels = labels.view(-1, 1).to(self.device)

            outputs = self.model.predict(inputs)

            loss = self.bce(outputs, labels)

            num_seqs += len(labels)
            running_loss += loss.item()

            if prediction is None:
                prediction = self.model.predict(inputs)
                allLabels = labels
            else:
                prediction = torch.cat((prediction, self.model.predict(inputs)), 0)
                allLabels = torch.cat((allLabels, labels), 0)


        print("Validation loss: "+str(running_loss/num_seqs))
        prediction = prediction.detach()
        prediction = prediction >= 0.5
        allLabels = allLabels.detach()
        allLabels = allLabels >= 0.5
        print("precision: " + str(met.precision_score(allLabels, prediction)))
        print("recall: " + str(met.recall_score(allLabels, prediction)))
        print("accuracy: " + str(met.accuracy_score(allLabels, prediction)))
        print("F1-score: " + str(met.f1_score(allLabels, prediction)))
        print("MCC: " + str(met.matthews_corrcoef(allLabels, prediction)))
        return running_loss / num_seqs

    def save_model(self):
        if self.save_path:
            torch.save(self.model.state_dict(), self.save_path)

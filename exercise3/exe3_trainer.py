import torch
from torch import nn
import torch.optim as optim
from typing import Any
from tqdm.auto import tqdm
import sklearn.metrics as met
from exe3_network import Model
import random
import h5py
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, List

import optuna
from optuna.trial import TrialState


class FancyDataset(Dataset):
    def __init__(self, labels: List[int], data: List[torch.Tensor]):
        self.data = data
        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return (self.data[index], self.labels[index])

    def __len__(self) -> int:
        return len(self.data)

def collate_paired_sequences(data: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([x.tolist() for x in list(zip(*data))[0]]),
        torch.tensor([[x] for x in list(zip(*data))[1]]).float()
    )

class Trainer:
    def __init__(self, model: Any, device: torch.device, save_path: str = 'hypopt_model.pth'):
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
            torch.save(
                {'model': self.model.state_dict(), 'config': self.model.get_config()}, self.save_path
            )

def getDataloaders():
    h5file = h5py.File('data/embedding.h5', 'r')
    X = h5file['X'][:]
    y = h5file['y'][:]
    h5file.close()

    torch.manual_seed(47)
    testing_idx = random.sample(range(0, 5883), 588) + random.sample(range(5883, 18650), 1277)

    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    for i in range(18650):
        if i in testing_idx:
            test_labels.append(y[i])
            test_data.append(torch.Tensor(X[i]))
        else:
            train_labels.append(y[i])
            train_data.append(torch.Tensor(X[i]))

    test_temp = list(zip(test_data, test_labels))
    train_temp = list(zip(train_data, train_labels))
    random.shuffle(test_temp)
    random.shuffle(train_temp)
    test_data, test_labels = zip(*test_temp)
    train_data, train_labels = zip(*train_temp)

    test_dataset = FancyDataset(test_labels, test_data)
    train_dataset = FancyDataset(train_labels, train_data)

    batch_size = 1028
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_paired_sequences)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_paired_sequences)

    return test_dataloader, train_dataloader


def objective(trial):

    num_epoch = 10
    device = torch.device('cpu')

    model = Model(trial).to(device) #

    sigmoid = nn.Sigmoid()
    bce_with_logits = nn.BCEWithLogitsLoss()
    bce = nn.BCELoss()

    val_dataloader, train_dataloader = getDataloaders()

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    model.train()
    for epoch in tqdm(range(num_epoch), desc='Train Epoch', position=0, leave=True, ascii=True):
        running_loss, num_seqs = 0.0, 0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            labels = labels.view(-1, 1).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = bce_with_logits(outputs, labels)
            loss.backward()
            optimizer.step()
            num_seqs += len(labels)

            running_loss += loss.item()

        optimizer.zero_grad()
        model.eval()
        running_loss, num_seqs = 0.0, 0
        prediction = None
        allLabels = None
        for i, data in enumerate(val_dataloader):
            inputs, labels = data
            labels = labels.view(-1, 1).to(device)

            outputs = model.predict(inputs)

            loss = bce(outputs, labels)

            num_seqs += len(labels)
            running_loss += loss.item()

            if prediction is None:
                prediction = model.predict(inputs)
                allLabels = labels
            else:
                prediction = torch.cat((prediction, model.predict(inputs)), 0)
                allLabels = torch.cat((allLabels, labels), 0)


    prediction = prediction.detach()
    prediction = prediction >= 0.5
    allLabels = allLabels.detach()
    allLabels = allLabels >= 0.5
    print("Validation loss: "+str(running_loss/num_seqs))
    print("precision: " + str(met.precision_score(allLabels, prediction)))
    print("recall: " + str(met.recall_score(allLabels, prediction)))
    print("accuracy: " + str(met.accuracy_score(allLabels, prediction)))
    print("F1-score: " + str(met.f1_score(allLabels, prediction)))
    print("MCC: " + str(met.matthews_corrcoef(allLabels, prediction)))

    loss = running_loss/num_seqs
    acc = met.accuracy_score(allLabels, prediction)
    trial.report(loss, epoch)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()


    return loss

def save_model(save_path, model):
    if save_path:
        torch.save(
            {'model': model.state_dict(), 'config': model.get_config()},
            save_path
        )


if __name__ == '__main__':
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    test_dataloader, train_dataloader = getDataloaders()
    
    model = Model(trial)
    trainer = Trainer(model, torch.device('cpu'))
    trainer.train(train_dataloader, 20)

    checkpoint = torch.load('hypopt_model.pth')
    model = Model(**checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()

    trainer = Trainer(model, device=torch.device('cpu'))
    trainer.validate(test_dataloader)


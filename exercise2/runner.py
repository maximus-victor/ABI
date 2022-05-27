import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, List
import random
from exe2_trainer import Trainer
from exe2_network import Model


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


h5file = h5py.File('data/embedding.h5', 'r')
X = h5file['X'][:]
y = h5file['y'][:]
h5file.close()

torch.manual_seed(47)

# class 1: 0-5882
# class 2: 5883-18649

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

model = Model()
trainer = Trainer(model, torch.device('cpu'))
trainer.train(train_dataloader, 10)
trainer.validate(test_dataloader)

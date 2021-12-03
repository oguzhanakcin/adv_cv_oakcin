import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MNISTClassifier(nn.Module):

    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.EmbeddingLearner = nn.Sequential(
            nn.Conv2d(1,16,3,padding=(1,1)),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(16,32,3,padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            nn.Conv2d(32, 32, 3, padding=(1, 1)),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(32, 32, 3, padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(7*7*32, 128),
            nn.ReLU(True),
        )
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input):
        x = self.EmbeddingLearner(input)
        flat_x = torch.flatten(x,1)
        embedding = self.fc1(flat_x)
        output = self.fc2(embedding)
        return embedding, output

class MNISTDataset(Dataset):
    def __init__(self,X,y,transform=None):
        self.data = X.clone().detach().float().reshape((-1, 1, 28, 28))
        self.label = y.clone().detach().int()
        self.transform = transform

    def __len__(self):
        return self.label.size(0)

    def __getitem__(self,index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        y = self.label[index]

        return (x,y)
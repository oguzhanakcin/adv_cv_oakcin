import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.EmbeddingLearner = nn.Sequential(
            nn.Linear(2, 7),
            nn.ReLU(True),
            nn.Linear(7, 8),
            nn.ReLU(True)

        )
        self.classifier = nn.Sequential(
            nn.Linear(8, 7))

    def forward(self, input):
        embedding = self.EmbeddingLearner(input)
        output = self.classifier(embedding)
        return embedding, output

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm1d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)

class GU(nn.Module):
    def __init__(self):
        super(GU, self).__init__()
        self.gu_prob = nn.Sequential(
            nn.Linear(15, 8),
            nn.ReLU(True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self,  emb,logit):
        x = torch.cat((emb,logit), 1)
        out = self.gu_prob(x)
        return out
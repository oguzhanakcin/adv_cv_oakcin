from utils import *
import torch
from torch.utils.data import Dataset
from SCOD_codes.nn_ood.distributions import CategoricalLogit
from SCOD_codes.nn_ood.posteriors.scod import SCOD
import torch.nn as nn

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

class Synthetic:

    def __init__(self, data, hyp,hypgen, device):

        self.device = device

        self.X_train, self.X_test, self.y_train, self.y_test = data[0], data[1], data[2], data[3]

        self.datasets = []
        self.models = []
        self.test_dataset = create_dataset(self.X_test, self.y_test)
        self.train_dataset = create_dataset(self.X_train, self.y_train)
        self.n_device = hypgen["n_device"]
        self.n_class = hypgen["n_class"]
        self.obs_clss = hypgen["observed_classes"]
        self.trainuncs = [[[] for i in range(self.n_class)] for j in range(self.n_device)]
        self.testuncs = [[[] for i in range(self.n_class)] for j in range(self.n_device)]

        for i in range(self.n_device):
            filt = [True if j in hypgen["base_classes"][i] else False for j in self.y_train]
            datas = create_dataset(self.X_train[filt], self.y_train[filt])
            self.datasets.append(datas)
            model_i = Classifier().to(device)
            model_i.load_state_dict(torch.load(hyp["weight_loc"]+ "basemodel"+str(i)+".pt"))
            self.models.append(model_i)

        self.train_datasets = [i.to(device) for i in self.datasets]
        self.dataloaders = [[] for i in range(self.n_device)]
        self.ind_classes = [[] for i in range(self.n_device)]
        self.ood_classes = [[] for i in range(self.n_device)]
        for i in range(self.n_device):
            for j in hypgen["desired_classes"][i]:
                if j not in hypgen["base_classes"][i]:
                    self.ood_classes[i].append(j)
                else:
                    self.ind_classes[i].append(j)

    def test_function(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1)
        data = torch.zeros((1, 2), dtype=torch.float).to(self.device)
        label = torch.zeros((1), dtype=int).to(self.device)
        confusion_matrices = [np.zeros((self.n_class, self.n_class), dtype=int) for i in range(self.n_device)]

        for n_i in range(self.n_device):
            self.models[n_i].eval()

            with torch.no_grad():
                for test_data in test_loader:

                    data[:, :] = test_data[:, :2]
                    label[:] = test_data[:, 2]

                    emb, out = self.models[n_i](data)

                    _, pred = torch.max(out.data, 1)

                    for i in range(self.n_class):
                        filt_i = (label == i)

                        pred_i = pred[filt_i]

                        for j in range(self.n_class):
                            filt_j = (pred_i == j)

                            nnum = sum(filt_j)

                            confusion_matrices[n_i][i, j] += nnum

            all_accs, ood_accs = acc_calc(confusion_matrices[n_i], self.n_class, self.ind_classes[n_i])
            print("Accuracy of All :", '{0:.3g}'.format(all_accs))
            print("Accuracy of OoD :", '{0:.3g}'.format(ood_accs))

    def save_ood_scores(self,name,out_loc):
        train_ood_unc = []
        train_ind_unc = []
        test_ood_unc = []
        test_ind_unc = []

        for i in range(self.n_device):
            for j in range(self.n_class):
                if j in self.ood_classes[i]:
                    train_ood_unc.extend(self.trainuncs[i][j])
                    test_ood_unc.extend(self.testuncs[i][j])
                else:
                    train_ind_unc.extend(self.trainuncs[i][j])
                    test_ind_unc.extend(self.testuncs[i][j])

        save_list(train_ood_unc, out_loc, name + "_train_ood_unc.npy")
        save_list(train_ind_unc, out_loc, name + "_train_ind_unc.npy")
        save_list(test_ood_unc, out_loc, name + "_test_ood_unc.npy")
        save_list(test_ind_unc, out_loc, name + "_test_ind_unc.npy")

class Softmax(Synthetic):

    def __init__(self, data, hyp,hypgen, device):
        super().__init__(data, hyp,hypgen, device)
        self.sofmax = nn.Softmax(dim=1)

    def ood_scores(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1)
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1)
        data = torch.zeros((1, 2), dtype=torch.float).to(self.device)
        label = torch.zeros((1), dtype=int).to(self.device)

        for i in range(self.n_device):

            self.models[i].eval()

            for test_data in test_loader:
                data[:, :] = test_data[:, :2]
                label[:] = test_data[:, 2]

                _, out = self.models[i](data)
                soft_out = self.sofmax(out)
                soft_max, _ = torch.max(soft_out, 1)
                soft_unc = 1 - soft_max

                self.testuncs[i][label[0]].append(soft_unc.tolist()[0])

            for train_data in train_loader:
                data[:, :] = train_data[:, :2]
                label[:] = train_data[:, 2]

                _, out = self.models[i](data)
                soft_out = self.sofmax(out)
                soft_max, _ = torch.max(soft_out, 1)
                soft_unc = 1 - soft_max

                self.trainuncs[i][label[0]].append(soft_unc.tolist()[0])

class Entropy(Synthetic):

    def __init__(self, data, hyp,hypgen, device):
        super().__init__(data, hyp,hypgen, device)
        self.sofmax = nn.Softmax(dim=1)

    def ood_scores(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1)
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1)
        data = torch.zeros((1, 2), dtype=torch.float).to(self.device)
        label = torch.zeros((1), dtype=int).to(self.device)

        for i in range(self.n_device):

            self.models[i].eval()

            for test_data in test_loader:
                data[:, :] = test_data[:, :2]
                label[:] = test_data[:, 2]

                _, out = self.models[i](data)
                soft_out = self.sofmax(out)
                log_outs = torch.log(soft_out)
                entr = torch.sum(-soft_out * log_outs, 1)

                self.testuncs[i][label[0]].append(entr.tolist()[0])

            for train_data in train_loader:
                data[:, :] = train_data[:, :2]
                label[:] = train_data[:, 2]

                _, out = self.models[i](data)
                soft_out = self.sofmax(out)
                log_outs = torch.log(soft_out)
                entr = torch.sum(-soft_out * log_outs, 1)

                self.trainuncs[i][label[0]].append(entr.tolist()[0])

class Scodselect(Synthetic):
    def __init__(self, data, hyp,hypgen, device):
        super().__init__(data, hyp,hypgen, device)
        self.dist_fam = CategoricalLogit().to(device)
        self.kwargs = {'num_samples': 64, 'num_eigs': 10, 'device': device, 'sketch_type': 'srft'}

    def prepro(self):
        prepro_datasets = [torch.utils.data.TensorDataset(train_dataset[:, :2],train_dataset[:, 2]) for train_dataset in self.train_datasets]
        self.scods = [SCOD(self.models[i],self.dist_fam,self.kwargs) for i in range(self.n_device)]

        for i in range(self.n_device):

            self.scods[i].process_dataset(prepro_datasets[i])
            data = torch.zeros((1, 2), dtype=torch.float).to(self.device)
            label = torch.zeros((1), dtype=int).to(self.device)

    def ood_scores(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1)
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1)
        data = torch.zeros((1, 2), dtype=torch.float).to(self.device)
        label = torch.zeros((1), dtype=int).to(self.device)

        for i in range(self.n_device):

            self.models[i].eval()

            for test_data in test_loader:
                data[:, :] = test_data[:, :2]
                label[:] = test_data[:, 2]

                _,_,unc = self.scods[i](data)

                self.testuncs[i][label[0]].append(unc.tolist()[0])

            for train_data in train_loader:
                data[:, :] = train_data[:, :2]
                label[:] = train_data[:, 2]

                _,_,unc = self.scods[i](data)

                self.trainuncs[i][label[0]].append(unc.tolist()[0])

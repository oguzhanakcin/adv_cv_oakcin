import torch
import torchvision.models as vsmodels
from torch.utils.data import Dataset
import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from SCOD_codes.nn_ood.distributions import CategoricalLogit
from SCOD_codes.nn_ood.posteriors.scod import SCOD
import torch.nn as nn
from utils import *

class RoadNet:
    def __init__(self, data, hyp,hypgen, device,dataset_loc):

        self.device = device

        self.X_train, self.y_train, self.X_test, self.y_test = data[0], data[1], data[2], data[3]

        self.datasets = []
        self.models = []
        self.test_dataset = create_RoadNet_dataset(self.X_test, self.y_test,dataset_loc)
        self.train_dataset = create_RoadNet_dataset(self.X_train, self.y_train,dataset_loc)

        self.model_size = hypgen["model_size"]

        self.n_device = hypgen["n_device"]
        self.n_class = hypgen["n_class"]
        self.obs_clss = hypgen["observed_classes"]
        self.trainuncs = [[[] for i in range(self.n_class)] for j in range(self.n_device)]
        self.testuncs = [[[] for i in range(self.n_class)] for j in range(self.n_device)]

        for i in range(self.n_device):
            X_train = [self.X_train[j] for j in range(len(self.y_train)) if
                       self.y_train[j] in hypgen["base_classes"][i]]
            y_train = [self.y_train[j] for j in range(len(self.y_train)) if
                       self.y_train[j] in hypgen["base_classes"][i]]
            datas = create_RoadNet_dataset(X_train, y_train,dataset_loc)
            self.datasets.append(datas)
            model_i = vsmodels.resnet18().to(device)
            model_i.fc = nn.Linear(in_features=512, out_features=4).to(device)
            model_i.load_state_dict(torch.load(hyp["weight_loc"]+ "smallbasemodel"+str(i)+".pt"))
            self.models.append(model_i)

        self.train_datasets = [i for i in self.datasets]
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
        data = torch.zeros((1, 3, 224, 224), dtype=torch.float).to(self.device)
        label = torch.zeros((1), dtype=int).to(self.device)
        confusion_matrices = [np.zeros((self.n_class, self.n_class), dtype=int) for i in range(self.n_device)]

        for n_i in range(self.n_device):
            self.models[n_i].eval()

            with torch.no_grad():
                for x, y in test_loader:

                    data[:] = x.reshape(data.shape)
                    label[:] = y

                    out = self.models[n_i](data)

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

        class_ids = ["sunny","overcast","rainy","snow"]

        for i in range(self.n_device):
            for j in range(self.n_class):
                if class_ids[j] in self.ood_classes[i]:
                    train_ood_unc.extend(self.trainuncs[i][j])
                    test_ood_unc.extend(self.testuncs[i][j])
                else:
                    train_ind_unc.extend(self.trainuncs[i][j])
                    test_ind_unc.extend(self.testuncs[i][j])

        save_list(train_ood_unc, out_loc, name + "_"+self.model_size+"_train_ood_unc.npy")
        save_list(train_ind_unc, out_loc, name + "_"+self.model_size+"_train_ind_unc.npy")
        save_list(test_ood_unc, out_loc, name + "_"+self.model_size+"_test_ood_unc.npy")
        save_list(test_ind_unc, out_loc, name + "_"+self.model_size+"_test_ind_unc.npy")


class RoadNetSoftmax(RoadNet):
    def __init__(self, data, hyp, hypgen, device,dataset_loc):
        super().__init__(data, hyp, hypgen, device,dataset_loc)

        self.sofmax = nn.Softmax(dim=1)

    def ood_scores(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1)
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1)
        data = torch.zeros((1, 3, 224, 224), dtype=torch.float).to(self.device)
        label = torch.zeros((1), dtype=int).to(self.device)

        for n_i in range(self.n_device):

            self.models[n_i].eval()

            for x,y in test_loader:
                data[:] = x.reshape(data.shape)
                label[:] = y

                out = self.models[n_i](data)
                soft_out = self.sofmax(out)
                soft_max, _ = torch.max(soft_out, 1)
                soft_unc = 1 - soft_max

                self.testuncs[n_i][label[0]].append(soft_unc.tolist()[0])

            for x, y in train_loader:
                data[:] = x.reshape(data.shape)
                label[:] = y

                out = self.models[n_i](data)
                soft_out = self.sofmax(out)
                soft_max, _ = torch.max(soft_out, 1)
                soft_unc = 1 - soft_max

                self.trainuncs[n_i][label[0]].append(soft_unc.tolist()[0])

class RoadNetEntropy(RoadNet):

    def __init__(self, data, hyp,hypgen, device,dataset_loc):
        super().__init__(data, hyp,hypgen, device,dataset_loc)
        self.sofmax = nn.Softmax(dim=1)

    def ood_scores(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1)
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1)
        data = torch.zeros((1, 3, 224, 224), dtype=torch.float).to(self.device)
        label = torch.zeros((1), dtype=int).to(self.device)

        for n_i in range(self.n_device):

            self.models[n_i].eval()

            for x,y in test_loader:
                data[:] = x.reshape(data.shape)
                label[:] = y

                out = self.models[n_i](data)
                soft_out = self.sofmax(out)
                log_outs = torch.log(soft_out)
                entr = torch.sum(-soft_out * log_outs, 1)

                self.testuncs[n_i][label[0]].append(entr.tolist()[0])

            for x, y in train_loader:
                data[:] = x.reshape(data.shape)
                label[:] = y

                out = self.models[n_i](data)
                soft_out = self.sofmax(out)
                log_outs = torch.log(soft_out)
                entr = torch.sum(-soft_out * log_outs, 1)

                self.trainuncs[n_i][label[0]].append(entr.tolist()[0])

class RoadNetScodselect(RoadNet):
    def __init__(self, data, hyp,hypgen, device,dataset_loc):
        super().__init__(data, hyp,hypgen, device,dataset_loc)
        self.dist_fam = CategoricalLogit().to(device)
        self.kwargs = {'num_samples': 604, 'num_eigs': 100, 'device': device, 'sketch_type': 'srft'}

    def prepro(self):
        self.scods = [SCOD(self.models[i],self.dist_fam,self.kwargs) for i in range(self.n_device)]
        for i in range(self.n_device):

            self.scods[i].process_dataset(self.train_datasets[i])

    def ood_scores(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1)
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1)
        data = torch.zeros((1, 3, 224, 224), dtype=torch.float).to(self.device)
        label = torch.zeros((1), dtype=int).to(self.device)

        for n_i in range(self.n_device):

            self.models[n_i].eval()

            for x,y in test_loader:
                data[:] = x.reshape(data.shape)
                label[:] = y

                _,unc = self.scods[n_i](data)


                self.testuncs[n_i][label[0]].append(unc.tolist()[0])

            for x, y in train_loader:
                data[:] = x.reshape(data.shape)
                label[:] = y

                _,unc = self.scods[n_i](data)

                self.trainuncs[n_i][label[0]].append(unc.tolist()[0])


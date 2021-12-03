import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as ds
import os, random
import torchvision.datasets as datasets
import torch
import torch.optim as optim
import torchvision.transforms as trfm
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.synt_models import *
from utils.MNIST_models import *
from SCOD_codes.nn_ood.distributions import CategoricalLogit
from SCOD_codes.nn_ood.posteriors.scod import SCOD
from iteround import saferound


def acc_calc(confusion_matrix,n_clss,in_clss):
    all_acc = 0
    ood_acc = 0

    for j in range(n_clss):
        all_acc += confusion_matrix[j, j] / (np.sum(confusion_matrix, axis=1)[j])
        if j not in in_clss:
            ood_acc += confusion_matrix[j, j] / (np.sum(confusion_matrix, axis=1)[j])

    all_acc /= n_clss
    ood_acc /= (n_clss - len(in_clss) + 1e-10)

    return all_acc, ood_acc

def test_function(model, test_dataset, device):

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    data = torch.zeros((1, 2), dtype=torch.float).to(device)
    label = torch.zeros((1), dtype=int).to(device)
    confusion_matrix = np.zeros((7, 7), dtype=int)
    model.eval()
    with torch.no_grad():
        for test_data in test_loader:

            data[:, :] = test_data[:, :2]
            label[:] = test_data[:, 2]

            emb, out = model(data)

            _, pred = torch.max(out.data, 1)

            for i in range(7):
                filt_i = (label == i)
                pred_i = pred[filt_i]
                for j in range(7):
                    filt_j = (pred_i == j)
                    nnum = sum(filt_j)
                    confusion_matrix[i, j] += nnum

    return confusion_matrix

def test_MNIST_function(model, test_dataset, device):

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    data = torch.zeros((1, 1, 28, 28), dtype=torch.float).to(device)
    label = torch.zeros((1), dtype=int).to(device)
    confusion_matrix = np.zeros((10, 10), dtype=int)
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:

            data[:] = x.reshape(data.shape)
            label[:] = y

            emb, out = model(data)

            _, pred = torch.max(out.data, 1)

            for i in range(10):
                filt_i = (label == i)
                pred_i = pred[filt_i]
                for j in range(10):
                    filt_j = (pred_i == j)
                    nnum = sum(filt_j)
                    confusion_matrix[i, j] += nnum

    return confusion_matrix

def train_model(model, losses, loss_fn, optimizer, num_epoch, data, label, dataloader, silent=True):
    model.train()

    if not silent:
        pbar = tqdm([i for i in range(num_epoch)], total=num_epoch)
    else:
        pbar = [i for i in range(num_epoch)]
    for epoch in pbar:
        for data_i in dataloader:
            data[:, :] = data_i[:, :2]
            label[:] = data_i[:, 2]

            model.zero_grad()

            emb, out = model(data)
            loss = loss_fn(out, label)
            losses.append(loss)

            loss.backward()

            optimizer.step()
        if not silent:
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
            pbar.set_description(s)
    return losses

def train_MNIST_model(model, losses, loss_fn, optimizer, num_epoch, data, label, dataloader, silent=True):
    model.train()

    if not silent:
        pbar = tqdm([i for i in range(num_epoch)], total=num_epoch)
    else:
        pbar = [i for i in range(num_epoch)]
    for epoch in pbar:
        for x,y in dataloader:
            data[:] = x.reshape(data.shape)
            label[:] = y

            model.zero_grad()

            emb, out = model(data)
            loss = loss_fn(out, label)
            losses.append(loss)

            loss.backward()

            optimizer.step()
        if not silent:
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
            pbar.set_description(s)
    return losses

# Generate Synthetic Dataset
def create_synt_dataset(save_dir, num_class, num_sample, train_ratio):
    X, y = ds.make_blobs(n_samples=num_sample * num_class, centers=num_class, n_features=2, random_state=10)
    X_max = X.max(axis=0)
    X_min = X.min(axis=0)
    X_diff = X_max - X_min
    Xn = 2 * ((X - X_min) / X_diff - 0.5)

    # Split it into train and test set

    X_train, X_test, y_train, y_test = train_test_split(Xn, y, train_size=train_ratio, stratify=y,random_state=1)

    isExist = os.path.exists(save_dir)

    if not isExist:
        os.makedirs(save_dir)

    np.save(save_dir + '/X_train.npy', X_train)
    np.save(save_dir + '/X_test.npy', X_test)
    np.save(save_dir + '/y_train.npy', y_train)
    np.save(save_dir + '/y_test.npy', y_test)

def load_MNIST_dataset(save_dir):
    mnist_trainset = datasets.MNIST(root=save_dir, train=True, download=True)
    mnist_testset = datasets.MNIST(root=save_dir, train=False, download=True)
    X_train = mnist_trainset.data
    y_train = mnist_trainset.targets
    X_test = mnist_testset.data
    y_test = mnist_testset.targets

    return X_train, y_train, X_test, y_test

def load_synt_dataset(dir, show_plots=False):
    X_train = np.load(dir + '/X_train.npy')
    X_test = np.load(dir + '/X_test.npy')
    y_train = np.load(dir + '/y_train.npy')
    y_test = np.load(dir + '/y_test.npy')

    if show_plots:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, ax=axs[0], palette="deep")
        axs[0].set_title("Train Set")
        sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, ax=axs[1], palette="deep")
        axs[1].set_title("Test Set")
        plt.show()

    return X_train, X_test, y_train, y_test

def cam_filter(classes, X_train, y_train):
    filt_cam = [True if i in classes else False for i in y_train]
    X_cam = X_train[filt_cam].tolist()
    y_cam = y_train[filt_cam].tolist()
    data_cam = [list(x) + [y_cam[i]] for i, x in enumerate(X_cam)]
    return data_cam

def cam_MNIST_filter(classes, X_train, y_train):
    filt_cam = [True if i in classes else False for i in y_train]
    X_cam = X_train[filt_cam].tolist()
    y_cam = y_train[filt_cam].tolist()
    data_cam =[(x,y_cam[i]) for i, x in enumerate(X_cam)]
    return data_cam

def create_dataset(X, y):
    data = torch.tensor(X, dtype=torch.float)
    label = torch.tensor(y, dtype=int)
    dataset = torch.zeros((data.shape[0], 3))
    dataset[:, :2] = data
    dataset[:, 2] = label
    return dataset

def create_dataloader(dataset, b_size, device):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=b_size, shuffle=True, drop_last=True, worker_init_fn=0)
    data = torch.zeros((b_size, 2), dtype=torch.float).to(device)
    label = torch.zeros((b_size), dtype=int).to(device)
    return dataloader, data, label

def create_MNIST_dataset(X,y):
    transform = trfm.Normalize((0.1307,), (0.3081,))
    dataset = MNISTDataset(X,y,transform)
    return dataset

def create_MNIST_dataloader(dataset, b_size, device):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=b_size, shuffle=True, drop_last=True, worker_init_fn=0)
    data = torch.zeros((b_size,1, 28,28), dtype=torch.float).to(device)
    label = torch.zeros((b_size), dtype=int).to(device)
    return dataloader, data, label

def save_list(i_list, dir, name):
    isExist = os.path.exists(dir)

    if not isExist:
        os.makedirs(dir)

    np_list = np.array(i_list)
    np.save(dir + "/" + name, np_list)

def find_order(a,x):
    lo = 0
    hi = len(a)

    while lo < hi :
        mid = (lo+hi)//2
        if x < a[mid]:
            hi = mid
        else:
            lo=mid+1

    return lo

class Synthetic:

    def __init__(self, data, hyp,hypgen, device):

        self.device = device

        self.X_train, self.X_test, self.y_train, self.y_test = data[0], data[1], data[2], data[3]

        self.datasets = []
        self.models = []
        self.optimizers = []
        self.lr_schedulers = []
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = hyp["lr"]
        self.test_dataset = create_dataset(self.X_test, self.y_test)
        self.n_device = hypgen["n_device"]
        self.n_class = hypgen["n_class"]
        self.obs_clss = hypgen["observed_classes"]
        for i in range(self.n_device):
            filt = [True if j in hypgen["base_classes"][i] else False for j in self.y_train]
            datas = create_dataset(self.X_train[filt], self.y_train[filt])
            self.datasets.append(datas)
            model_i = Classifier().to(device)
            model_i.load_state_dict(torch.load(hyp["weight_loc"]+ "basemodel"+str(i)+".pt"))
            self.models.append(model_i)
            opts = optim.SGD(self.models[i].parameters(), lr=self.lr, weight_decay=0.01)
            self.optimizers.append(opts)
            lr_sch = lr_scheduler.ExponentialLR(self.optimizers[i], gamma=0.9, last_epoch=-1)
            self.lr_schedulers.append(lr_sch)

        self.T = hyp["T"]
        self.N = hyp["N"]
        self.cache_size = hyp["cache_size"]
        self.round_numb = hyp["round_num"]
        self.num_epoch = hyp["num_epoch"]
        self.b_size = hyp["b_size"]
        self.losses = [[] for i in range(self.n_device)]
        self.nimgs = [[] for i in range(self.n_device)]
        self.avgs = [[] for i in range(self.n_device)]
        self.prevs = [0 for i in range(self.n_device)]
        self.all_accs = [[] for i in range(self.n_device)]
        self.ood_accs = [[] for i in range(self.n_device)]

        self.data_cams = [cam_filter(hypgen["observed_classes"][i], self.X_train,self.y_train) for i in range(self.n_device)]

        self.data = torch.zeros((self.b_size, 2), dtype=torch.float).to(device)

        self.label = torch.zeros((self.b_size), dtype=int).to(device)

        self.train_datasets = [i.to(device) for i in self.datasets]
        self.dataloaders = [[] for i in range(self.n_device)]
        self.ood_classes = [[] for i in range(self.n_device)]
        for i in range(self.n_device):
            for j in hypgen["desired_classes"][i]:
                if j not in hypgen["base_classes"][i]:
                    self.ood_classes[i].append(j)

    def img_generate(self):

        self.imgs = [random.sample(self.data_cams[i],self.cache_size) for i in range(self.n_device)]

    def util_calculate(self):

        utils = [0 for j in range(self.n_device)]

        for i in range(self.n_device):
            for j in range(self.n_device):
                for rc in self.caches[i]:
                    if rc[2] in self.ood_classes[j]:
                        utils[i] += 1
            utils[i] = utils[i]/self.n_device

        for i in range(self.n_device):
            self.prevs[i] += utils[i]
            self.nimgs[i].append(self.prevs[i])
            self.avgs[i].append(utils[i])

    def update_trainset(self):
        caches = []
        for i in range(self.n_device):
            caches += self.caches[i]
        cached_dataset = torch.tensor(caches).to(self.device)
        for i in range(self.n_device):
            self.train_datasets[i] =  torch.cat((self.train_datasets[i], cached_dataset))

    def create_dataloaders(self):

        for i in range(self.n_device):
            self.dataloaders[i] = torch.utils.data.DataLoader(self.train_datasets[i], batch_size=self.b_size, shuffle=True,
                                                       drop_last=True,
                                                       worker_init_fn=0)

    def train_model(self, silent=True):

        for n_i in range(self.n_device):
            self.models[n_i].train()

            if not silent:
                pbar = tqdm([i for i in range(self.num_epoch)], total=self.num_epoch)
            else:
                pbar = [i for i in range(self.num_epoch)]

            for epoch in pbar:
                dataiter = iter(self.dataloaders[n_i])

                for i in range(len(dataiter)):
                    data_i = dataiter.next()

                    self.data[:, :] = data_i[:, :2]
                    self.label[:] = data_i[:, 2]

                    self.models[n_i].zero_grad()

                    emb, out = self.models[n_i](self.data)

                    loss = self.loss_fn(out, self.label)

                    loss.backward()

                    self.optimizers[n_i].step()

                if not silent:
                    mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                    s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
                    pbar.set_description(s)

    def test_function(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1)
        data = torch.zeros((1, 2), dtype=torch.float).to(self.device)
        label = torch.zeros((1), dtype=int).to(self.device)
        confusion_matrices = [np.zeros((7, 7), dtype=int) for i in range(self.n_device)]

        for n_i in range(self.n_device):
            self.models[n_i].eval()

            with torch.no_grad():
                for test_data in test_loader:

                    data[:, :] = test_data[:, :2]
                    label[:] = test_data[:, 2]

                    emb, out = self.models[n_i](data)

                    _, pred = torch.max(out.data, 1)

                    for i in range(7):
                        filt_i = (label == i)

                        pred_i = pred[filt_i]

                        for j in range(7):
                            filt_j = (pred_i == j)

                            nnum = sum(filt_j)

                            confusion_matrices[n_i][i, j] += nnum


        self.confusion_matrices = confusion_matrices

    def acc_calc(self):

        for i in range(self.n_device):
            all_acc = 0
            ood_acc = 0
            for j in range(self.n_class):
                all_acc += self.confusion_matrices[i][j, j] / (np.sum(self.confusion_matrices[i], axis=1)[j])
                if j in self.ood_classes[i]:
                    ood_acc += self.confusion_matrices[i][j, j] / (np.sum(self.confusion_matrices[i], axis=1)[j])

            all_acc /= self.n_class
            ood_acc /= len(self.ood_classes[i])

            self.all_accs[i].append(all_acc)
            self.ood_accs[i].append(ood_acc)

    def save_sim_data(self, name, out_loc):
        self.nimg = [0 for i in range(self.round_numb)]
        self.all_acc = [0 for i in range(self.round_numb)]
        self.ood_acc = [0 for i in range(self.round_numb)]

        for i in range(self.n_device):
            for j in range(self.round_numb):
                self.nimg[j] = self.nimgs[i][j]/self.n_device
                self.all_acc[j] = self.all_accs[i][j]/self.n_device
                self.ood_acc[j] = self.ood_accs[i][j] / self.n_device

        save_list(self.nimg, out_loc, name + "_nimg.npy")
        save_list(self.all_acc, out_loc, name + "_all_acc.npy")
        save_list(self.ood_acc, out_loc, name + "_ood_acc.npy")

class Random(Synthetic):
    def __init__(self, data, hyp, hypgen,device):
        super().__init__(data, hyp,hypgen, device)

    def cache_generate(self):
        self.caches = []

        for i in range(self.n_device):
            self.caches.append(random.sample(self.imgs[i], self.T))

class Oracle(Synthetic):
    def __init__(self, data, hyp,hypgen, device):
        super().__init__(data, hyp,hypgen, device)

    def cache_generate(self):
        self.caches = []
        util_class = [0 for i in range(self.n_class)]
        seen_class = [0 for i in range(self.n_class)]
        for i in range(self.n_device):
            for j in self.ood_classes[i]:
                util_class[j] += 1/self.n_device
            for j in self.obs_clss[i]:
                seen_class[j] += 1

        for i in range(self.n_device):
            cache_ratio = [util_class[o]/seen_class[o] for o in self.obs_clss[i]]
            cache_ratio = [self.T*j/sum(cache_ratio) for j in cache_ratio]
            cache_numb = saferound(cache_ratio,places=0)
            cache_numb = [int(c) for c in cache_numb]
            cc = []
            for j in range(len(self.obs_clss[i])):
                caches_clss = [list(c) for c in self.imgs[i] if c[2] == self.obs_clss[i][j]]
                cc = cc + caches_clss[:cache_numb[j]]
            self.caches.append(cc)

class Softmax(Synthetic):

    def __init__(self, data, hyp,hypgen, device):
        super().__init__(data, hyp,hypgen, device)

        self.lsofmax = nn.LogSoftmax(dim=1)

    def cache_generate(self):
        self.caches = []

        for i in range(self.n_device):
            X_cam = torch.tensor([j[:2] for j in self.imgs[i]]).to(self.device)
            self.models[i].eval()
            with torch.no_grad():
                _, out = self.models[i](X_cam)
            lsoft_out = self.lsofmax(out)
            lsof_maxs, _ = torch.max(lsoft_out, 1)
            _, ood_ind = torch.topk(-lsof_maxs, self.T)
            self.caches.append([self.imgs[i][j] for j in list(ood_ind)])

class Entropy(Synthetic):

    def __init__(self, data, hyp,hypgen, device):
        super().__init__(data, hyp,hypgen, device)
        self.sofmax = nn.Softmax(dim=1)

    def cache_generate(self):
        self.caches = []

        for i in range(self.n_device):
            X_cam = torch.tensor([j[:2] for j in self.imgs[i]]).to(self.device)
            self.models[i].eval()
            with torch.no_grad():
                _, out = self.models[i](X_cam)
            soft_out = self.sofmax(out)
            log_outs = torch.log(soft_out)
            entr = torch.sum(-soft_out * log_outs, 1)

            _, ood_ind = torch.topk(entr, self.T)
            self.caches.append([self.imgs[i][j] for j in list(ood_ind)])

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

    def cache_generate(self):
        self.caches = []

        for n_i in range(self.n_device):
            X_cam = torch.tensor([i[:2] for i in self.imgs[n_i]]).to(self.device)

            self.models[n_i].eval()

            _, _, unc = self.scods[n_i](X_cam)

            _, ood_ind = torch.topk(unc, self.T, 0)


            self.caches.append([self.imgs[n_i][i] for i in list(ood_ind)])

    def ood_scores(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1)
        data = torch.zeros((1, 2), dtype=torch.float).to(self.device)
        label = torch.zeros((1), dtype=int).to(self.device)

        self.uncs = [[[] for i in range(self.n_class)] for j in range(self.n_device)]

        for i in range(self.n_device):

            self.models[i].eval()

            for test_data in test_loader:
                data[:, :] = test_data[:, :2]
                label[:] = test_data[:, 2]

                _,_,unc = self.scods[i](data)

                self.uncs[i][label[0]].append(unc.tolist()[0])

    def save_ood_scores(self,name,out_loc):
        self.ood_unc = []
        self.ind_unc = []

        for i in range(self.n_device):
            for j in range(self.n_class):
                if j in self.ood_classes[i]:
                    self.ood_unc.extend(self.uncs[i][j])
                else:
                    self.ind_unc.extend(self.uncs[i][j])

        save_list(self.ood_unc, out_loc,name + "_ood_unc.npy")
        save_list(self.ind_unc, out_loc, name + "_ind_unc.npy")

class Scodgu(Scodselect):
    def __init__(self, data, hyp, device):
        super().__init__(data, hyp, device)

        self.gu_model1 = GUScod().to(device)
        self.gu_model1.apply(init_weights)
        self.gu_model2 = GUScod().to(device)
        self.gu_model2.apply(init_weights)
        self.gu_model3 = GUScod().to(device)
        self.gu_model3.apply(init_weights)

        self.gu_loss_fn = nn.BCELoss()
        self.gu_lr = hyp["gu_lr"]

        self.gu_optimizer1 = optim.SGD(self.gu_model1.parameters(), lr=self.gu_lr, weight_decay=0.01)
        self.gu_lr_scheduler1 = lr_scheduler.ExponentialLR(self.gu_optimizer1, gamma=0.9, last_epoch=-1)

        self.gu_optimizer2 = optim.SGD(self.gu_model2.parameters(), lr=self.gu_lr, weight_decay=0.01)
        self.gu_lr_scheduler2 = lr_scheduler.ExponentialLR(self.gu_optimizer2, gamma=0.9, last_epoch=-1)

        self.gu_optimizer3 = optim.SGD(self.gu_model3.parameters(), lr=self.gu_lr, weight_decay=0.01)
        self.gu_lr_scheduler3 = lr_scheduler.ExponentialLR(self.gu_optimizer3, gamma=0.9, last_epoch=-1)

        self.gu_losses1, self.gu_losses2, self.gu_losses3 = [], [], []
        self.gu_data1 = torch.zeros((self.T, 2), dtype=torch.float).to(device)
        self.gu_label1 = torch.zeros((self.T, 1), dtype=torch.float).to(device)
        self.gu_data2 = torch.zeros((self.T, 2), dtype=torch.float).to(device)
        self.gu_label2 = torch.zeros((self.T, 1), dtype=torch.float).to(device)
        self.gu_data3 = torch.zeros((self.T, 2), dtype=torch.float).to(device)
        self.gu_label3 = torch.zeros((self.T, 1), dtype=torch.float).to(device)
        self.gu_data4 = torch.zeros((self.T, 2), dtype=torch.float).to(device)
        self.gu_label4 = torch.zeros((self.T, 1), dtype=torch.float).to(device)
        self.gu_data5 = torch.zeros((self.T, 2), dtype=torch.float).to(device)
        self.gu_label5 = torch.zeros((self.T, 1), dtype=torch.float).to(device)
        self.gu_data6 = torch.zeros((self.T, 2), dtype=torch.float).to(device)
        self.gu_label6 = torch.zeros((self.T, 1), dtype=torch.float).to(device)
        self.gu_data7 = torch.zeros((self.T, 2), dtype=torch.float).to(device)
        self.gu_label7 = torch.zeros((self.T, 1), dtype=torch.float).to(device)
        self.gu_data8 = torch.zeros((self.T, 2), dtype=torch.float).to(device)
        self.gu_label8 = torch.zeros((self.T, 1), dtype=torch.float).to(device)
        self.gu_data9 = torch.zeros((self.T, 2), dtype=torch.float).to(device)
        self.gu_label9 = torch.zeros((self.T, 1), dtype=torch.float).to(device)

    def cache_generate(self):
        self.caches = []

        X_cam1 = torch.tensor([i[:2] for i in self.imgs1]).to(self.device)
        X_cam2 = torch.tensor([i[:2] for i in self.imgs2]).to(self.device)
        X_cam3 = torch.tensor([i[:2] for i in self.imgs3]).to(self.device)

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()

        self.gu_model1.eval()
        self.gu_model2.eval()
        self.gu_model3.eval()

        emb1, out1, unc1 = self.scod1(X_cam1)
        emb2, out2, unc2 = self.scod2(X_cam2)
        emb3, out3, unc3 = self.scod3(X_cam3)

        score1 = self.gu_model1(emb1, out1, torch.reshape(unc1, (-1, 1)).to(self.device))
        score2 = self.gu_model2(emb2, out2, torch.reshape(unc2, (-1, 1)).to(self.device))
        score3 = self.gu_model3(emb3, out3, torch.reshape(unc3, (-1, 1)).to(self.device))

        scores_11 = score1[:, 0].cpu().detach().numpy()
        scores_12 = score1[:, 1].cpu().detach().numpy()
        scores_13 = score1[:, 2].cpu().detach().numpy()
        scores_21 = score2[:, 0].cpu().detach().numpy()
        scores_22 = score2[:, 1].cpu().detach().numpy()
        scores_23 = score2[:, 2].cpu().detach().numpy()
        scores_31 = score3[:, 0].cpu().detach().numpy()
        scores_32 = score3[:, 1].cpu().detach().numpy()
        scores_33 = score3[:, 2].cpu().detach().numpy()

        ind11 = np.random.choice(len(scores_11), self.T, p=scores_11 / scores_11.sum(), replace=False)
        ind12 = np.random.choice(len(scores_12), self.T, p=scores_12 / scores_12.sum(), replace=False)
        ind13 = np.random.choice(len(scores_13), self.T, p=scores_13 / scores_13.sum(), replace=False)
        ind21 = np.random.choice(len(scores_21), self.T, p=scores_21 / scores_21.sum(), replace=False)
        ind22 = np.random.choice(len(scores_22), self.T, p=scores_22 / scores_22.sum(), replace=False)
        ind23 = np.random.choice(len(scores_23), self.T, p=scores_23 / scores_23.sum(), replace=False)
        ind31 = np.random.choice(len(scores_31), self.T, p=scores_31 / scores_31.sum(), replace=False)
        ind32 = np.random.choice(len(scores_32), self.T, p=scores_32 / scores_32.sum(), replace=False)
        ind33 = np.random.choice(len(scores_33), self.T, p=scores_33 / scores_33.sum(), replace=False)

        self.caches.append([self.imgs1[i] for i in list(ind11)])
        self.caches.append([self.imgs1[i] for i in list(ind12)])
        self.caches.append([self.imgs1[i] for i in list(ind13)])

        self.caches.append([self.imgs2[i] for i in list(ind21)])
        self.caches.append([self.imgs2[i] for i in list(ind22)])
        self.caches.append([self.imgs2[i] for i in list(ind23)])

        self.caches.append([self.imgs3[i] for i in list(ind31)])
        self.caches.append([self.imgs3[i] for i in list(ind32)])
        self.caches.append([self.imgs3[i] for i in list(ind33)])

    def gu_unc_scores(self):

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()

        self.uncs1 = []
        self.uncs2 = []
        self.uncs3 = []

        gu_loader1 = torch.utils.data.DataLoader(self.train_dataset1, batch_size=self.T)
        gu_loader2 = torch.utils.data.DataLoader(self.train_dataset2, batch_size=self.T)
        gu_loader3 = torch.utils.data.DataLoader(self.train_dataset3, batch_size=self.T)

        dataiter1 = iter(gu_loader1)
        dataiter2 = iter(gu_loader2)
        dataiter3 = iter(gu_loader3)

        for i in range(len(dataiter1)):

            data1_i = dataiter1.next()
            data2_i = dataiter2.next()
            data3_i = dataiter3.next()

            self.gu_data1[:, :] = data1_i[:, :2]
            self.gu_data2[:, :] = data2_i[:, :2]
            self.gu_data3[:, :] = data3_i[:, :2]

            _, _, unc1 = self.scod1(self.gu_data1)
            _, _, unc2 = self.scod2(self.gu_data2)
            _, _, unc3 = self.scod3(self.gu_data3)

            self.uncs1.extend(unc1.tolist())
            self.uncs2.extend(unc2.tolist())
            self.uncs3.extend(unc3.tolist())

        self.uncs1.sort()
        self.uncs2.sort()
        self.uncs3.sort()

    def gu_train(self,i):

        self.gu_model1.train()
        self.gu_model2.train()
        self.gu_model3.train()

        c11 = torch.tensor(self.caches[0])[:, :2].to(self.device)
        c12 = torch.tensor(self.caches[1])[:, :2].to(self.device)
        c13 = torch.tensor(self.caches[2])[:, :2].to(self.device)
        c21 = torch.tensor(self.caches[3])[:, :2].to(self.device)
        c22 = torch.tensor(self.caches[4])[:, :2].to(self.device)
        c23 = torch.tensor(self.caches[5])[:, :2].to(self.device)
        c31 = torch.tensor(self.caches[6])[:, :2].to(self.device)
        c32 = torch.tensor(self.caches[7])[:, :2].to(self.device)
        c33 = torch.tensor(self.caches[8])[:, :2].to(self.device)

        _, _, unc1 = self.scod1(torch.cat((c11, c21, c31)))
        _, _, unc2 = self.scod1(torch.cat((c12, c22, c32)))
        _, _, unc3 = self.scod1(torch.cat((c13, c23, c33)))

        unc1list = unc1.tolist()
        unc2list = unc2.tolist()
        unc3list = unc3.tolist()

        unc1ord = [find_order(self.uncs1, i)/len(self.uncs1) for i in unc1list]
        unc2ord = [find_order(self.uncs2, i)/len(self.uncs2) for i in unc2list]
        unc3ord = [find_order(self.uncs3, i)/len(self.uncs3) for i in unc3list]

        lc11 = torch.tensor(unc1ord[:self.T]).to(self.device)
        lc12 = torch.tensor(unc2ord[:self.T]).to(self.device)
        lc13 = torch.tensor(unc3ord[:self.T]).to(self.device)
        lc21 = torch.tensor(unc1ord[self.T:2*self.T]).to(self.device)
        lc22 = torch.tensor(unc2ord[self.T:2*self.T]).to(self.device)
        lc23 = torch.tensor(unc3ord[self.T:2*self.T]).to(self.device)
        lc31 = torch.tensor(unc1ord[2*self.T:]).to(self.device)
        lc32 = torch.tensor(unc2ord[2*self.T:]).to(self.device)
        lc33 = torch.tensor(unc3ord[2*self.T:]).to(self.device)

        cl11 = torch.cat((c11, torch.reshape(lc11, (-1, 1))), 1)
        cl12 = torch.cat((c12, torch.reshape(lc12, (-1, 1))), 1)
        cl13 = torch.cat((c13, torch.reshape(lc13, (-1, 1))), 1)
        cl21 = torch.cat((c21, torch.reshape(lc21, (-1, 1))), 1)
        cl22 = torch.cat((c22, torch.reshape(lc22, (-1, 1))), 1)
        cl23 = torch.cat((c23, torch.reshape(lc23, (-1, 1))), 1)
        cl31 = torch.cat((c31, torch.reshape(lc31, (-1, 1))), 1)
        cl32 = torch.cat((c32, torch.reshape(lc32, (-1, 1))), 1)
        cl33 = torch.cat((c33, torch.reshape(lc33, (-1, 1))), 1)

        if i ==0:
            self.cl11s = cl11
            self.cl12s = cl12
            self.cl13s = cl13
            self.cl21s = cl21
            self.cl22s = cl22
            self.cl23s = cl23
            self.cl31s = cl31
            self.cl32s = cl32
            self.cl33s = cl33

        else:
            self.cl11s = torch.cat((self.cl11s, cl11))
            self.cl12s = torch.cat((self.cl12s, cl12))
            self.cl13s = torch.cat((self.cl13s, cl13))
            self.cl21s = torch.cat((self.cl21s, cl21))
            self.cl22s = torch.cat((self.cl22s, cl22))
            self.cl23s = torch.cat((self.cl23s, cl23))
            self.cl31s = torch.cat((self.cl31s, cl31))
            self.cl32s = torch.cat((self.cl32s, cl32))
            self.cl33s = torch.cat((self.cl33s, cl33))

        dataloader1 = torch.utils.data.DataLoader(self.cl11s, batch_size=self.T, shuffle=True,drop_last=True,
                                                  worker_init_fn=0)
        dataloader2 = torch.utils.data.DataLoader(self.cl12s, batch_size=self.T, shuffle=True, drop_last=True,
                                                  worker_init_fn=0)
        dataloader3 = torch.utils.data.DataLoader(self.cl13s, batch_size=self.T, shuffle=True, drop_last=True,
                                                  worker_init_fn=0)
        dataloader4 = torch.utils.data.DataLoader(self.cl21s, batch_size=self.T, shuffle=True, drop_last=True,
                                                  worker_init_fn=0)
        dataloader5 = torch.utils.data.DataLoader(self.cl22s, batch_size=self.T, shuffle=True, drop_last=True,
                                                  worker_init_fn=0)
        dataloader6 = torch.utils.data.DataLoader(self.cl23s, batch_size=self.T, shuffle=True, drop_last=True,
                                                  worker_init_fn=0)
        dataloader7 = torch.utils.data.DataLoader(self.cl31s, batch_size=self.T, shuffle=True, drop_last=True,
                                                  worker_init_fn=0)
        dataloader8 = torch.utils.data.DataLoader(self.cl32s, batch_size=self.T, shuffle=True, drop_last=True,
                                                  worker_init_fn=0)
        dataloader9 = torch.utils.data.DataLoader(self.cl33s, batch_size=self.T, shuffle=True, drop_last=True,
                                                  worker_init_fn=0)

        pbar = [i for i in range(self.num_epoch)]

        for epoch in pbar:
            dataiter1 = iter(dataloader1)
            dataiter2 = iter(dataloader2)
            dataiter3 = iter(dataloader3)
            dataiter4 = iter(dataloader4)
            dataiter5 = iter(dataloader5)
            dataiter6 = iter(dataloader6)
            dataiter7 = iter(dataloader7)
            dataiter8 = iter(dataloader8)
            dataiter9 = iter(dataloader9)

            for i in range(len(dataiter1)):
                data1_i = dataiter1.next()
                data2_i = dataiter2.next()
                data3_i = dataiter3.next()
                data4_i = dataiter4.next()
                data5_i = dataiter5.next()
                data6_i = dataiter6.next()
                data7_i = dataiter7.next()
                data8_i = dataiter8.next()
                data9_i = dataiter9.next()

                self.gu_data1[:, :] = data1_i[:,:2]
                self.gu_data2[:, :] = data2_i[:, :2]
                self.gu_data3[:, :] = data3_i[:, :2]
                self.gu_data4[:, :] = data4_i[:, :2]
                self.gu_data5[:, :] = data5_i[:, :2]
                self.gu_data6[:, :] = data6_i[:, :2]
                self.gu_data7[:, :] = data7_i[:, :2]
                self.gu_data8[:, :] = data8_i[:, :2]
                self.gu_data9[:, :] = data9_i[:, :2]

                self.gu_label1[:, 0] = data1_i[:, 2]
                self.gu_label2[:, 0] = data2_i[:, 2]
                self.gu_label3[:, 0] = data3_i[:, 2]
                self.gu_label4[:, 0] = data4_i[:, 2]
                self.gu_label5[:, 0] = data5_i[:, 2]
                self.gu_label6[:, 0] = data6_i[:, 2]
                self.gu_label7[:, 0] = data7_i[:, 2]
                self.gu_label8[:, 0] = data8_i[:, 2]
                self.gu_label9[:, 0] = data9_i[:, 2]

                self.gu_model1.zero_grad()
                self.gu_model2.zero_grad()
                self.gu_model3.zero_grad()

                emb1, out1, unc1 = self.scod1(self.gu_data1)
                emb2, out2, unc2 = self.scod1(self.gu_data2)
                emb3, out3, unc3 = self.scod1(self.gu_data3)
                emb4, out4, unc4 = self.scod2(self.gu_data4)
                emb5, out5, unc5 = self.scod2(self.gu_data5)
                emb6, out6, unc6 = self.scod2(self.gu_data6)
                emb7, out7, unc7 = self.scod3(self.gu_data7)
                emb8, out8, unc8 = self.scod3(self.gu_data8)
                emb9, out9, unc9 = self.scod3(self.gu_data9)


                gu_out1 = self.gu_model1(emb1, out1, torch.reshape(unc1, (-1, 1)).to(self.device))
                gu_out2 = self.gu_model1(emb2, out2, torch.reshape(unc2, (-1, 1)).to(self.device))
                gu_out3 = self.gu_model1(emb3, out3, torch.reshape(unc3, (-1, 1)).to(self.device))
                gu_out4 = self.gu_model2(emb4, out4, torch.reshape(unc4, (-1, 1)).to(self.device))
                gu_out5 = self.gu_model2(emb5, out5, torch.reshape(unc5, (-1, 1)).to(self.device))
                gu_out6 = self.gu_model2(emb6, out6, torch.reshape(unc6, (-1, 1)).to(self.device))
                gu_out7 = self.gu_model3(emb7, out7, torch.reshape(unc7, (-1, 1)).to(self.device))
                gu_out8 = self.gu_model3(emb8, out8, torch.reshape(unc8, (-1, 1)).to(self.device))
                gu_out9 = self.gu_model3(emb9, out9, torch.reshape(unc9, (-1, 1)).to(self.device))

                loss1 = self.gu_loss_fn(torch.reshape(gu_out1[:,0], (-1, 1)).to(self.device),self.gu_label1)
                loss2 = self.gu_loss_fn(torch.reshape(gu_out2[:,1], (-1, 1)).to(self.device), self.gu_label2)
                loss3 = self.gu_loss_fn(torch.reshape(gu_out3[:,2], (-1, 1)).to(self.device), self.gu_label3)

                loss4 = self.gu_loss_fn(torch.reshape(gu_out4[:,0], (-1, 1)).to(self.device), self.gu_label4)
                loss5 = self.gu_loss_fn(torch.reshape(gu_out5[:,1], (-1, 1)).to(self.device), self.gu_label5)
                loss6 = self.gu_loss_fn(torch.reshape(gu_out6[:,2], (-1, 1)).to(self.device), self.gu_label6)

                loss7 = self.gu_loss_fn(torch.reshape(gu_out7[:,0], (-1, 1)).to(self.device), self.gu_label7)
                loss8 = self.gu_loss_fn(torch.reshape(gu_out8[:,1], (-1, 1)).to(self.device), self.gu_label8)
                loss9 = self.gu_loss_fn(torch.reshape(gu_out9[:,2], (-1, 1)).to(self.device), self.gu_label9)

                losss1 = loss1 + loss2 + loss3
                losss2 = loss4 + loss5 + loss6
                losss3 = loss7 + loss8 + loss9

                self.gu_losses1.append(losss1)
                self.gu_losses2.append(losss2)
                self.gu_losses3.append(losss3)

                losss1.backward()
                losss2.backward()
                losss3.backward()

                self.gu_optimizer1.step()
                self.gu_optimizer2.step()
                self.gu_optimizer3.step()

class GUSampler(Synthetic):
    def __init__(self, data, hyp,hypgen, device):
        super().__init__(data,hyp,hypgen,device)

        self.gu_models = []
        self.gu_optimizers = []
        self.gu_lr_schedulers = []
        self.gu_lr = hyp["gu_lr"]
        self.gu_loss_fn = nn.BCELoss()
        self.disc_loss_fn = nn.BCELoss()
        self.disc_models = []
        self.disc_optimizers = []
        self.disc_lr_schedulers = []
        self.disc_lr = hyp["gu_lr"]

        for i in range(self.n_device):
            gu_model = GU().to(device)
            gu_model.apply(init_weights)
            self.gu_models.append(gu_model)
            optimizer = optim.SGD(self.gu_models[i].parameters(), lr=self.gu_lr, weight_decay=0.01)
            self.gu_optimizers.append(optimizer)
            gu_lr_scheduler = lr_scheduler.ExponentialLR(self.gu_optimizers[i], gamma=0.9, last_epoch=-1)
            self.gu_lr_schedulers.append(gu_lr_scheduler)
            disc_model = Discriminator().to(device)
            disc_model.apply(init_weights)
            self.disc_models.append(disc_model)
            optimizer = optim.SGD(self.disc_models[i].parameters(), lr=self.disc_lr, weight_decay=0.01)
            self.disc_optimizers.append(optimizer)
            disc_lr_scheduler = lr_scheduler.ExponentialLR(self.disc_optimizers[i], gamma=0.9, last_epoch=-1)
            self.disc_lr_schedulers.append(disc_lr_scheduler)

        self.sofmax = nn.Softmax(dim=1)

        self.gu_losses = [[] for i in range(self.n_device)]
        self.disc_losses = [[] for i in range(self.n_device)]
        self.gu_data = torch.zeros((self.T, 2), dtype=torch.float).to(device)
        self.gu_label = torch.zeros((self.T, 1), dtype=torch.float).to(device)
        self.disc_data = torch.zeros((self.T, 2), dtype=torch.float).to(device)
        self.disc_label = torch.zeros((self.T, 1), dtype=torch.float).to(device)

    def cache_generate(self):

        self.caches = []

        for i in range(self.n_device):
            X_cam = torch.tensor([j[:2] for j in self.imgs[i]]).to(self.device)
            self.models[i].eval()

            self.gu_models[i].eval()

            emb, out = self.models[i](X_cam)

            score = self.gu_models[i](emb, out)

            scores = score.cpu().detach().numpy()

            ind = np.random.choice(len(scores), self.T, p=scores[:,0] / scores.sum(), replace=False)

            self.caches.append([self.imgs[i][j] for j in list(ind)])

    def gu_train(self):

        for i in range(self.n_device):

            self.gu_models[i].train()
            self.models[i].eval()
            self.disc_models[i].eval()

            c_1 = torch.tensor(self.caches[0])[:, :2].to(self.device)
            c_2 = torch.tensor(self.caches[1])[:, :2].to(self.device)
            if i == 0
                l_1 = torch.tensor([1 for i in range(self.T)]).to(self.device)
                l_2 = torch.tensor([0 for i in range(self.T)]).to(self.device)
            else:
                l_1 = torch.tensor([0 for i in range(self.T)]).to(self.device)
                l_2 = torch.tensor([1 for i in range(self.T)]).to(self.device)


            c_i = torch.cat((c_1, c_2), 0)
            l_i = torch.cat((l_1, l_2), 0)

            cl_i = torch.cat((c_i, torch.reshape(l_i, (-1, 1))), 1)

            dataloader = torch.utils.data.DataLoader(cl_i, batch_size=self.T, shuffle=True, drop_last=True,
                                                  worker_init_fn=0)
            pbar = [i for i in range(20)]

            for epoch in pbar:
                dataiter = iter(dataloader)

                for k in range(len(dataiter)):
                    data_i = dataiter.next()

                    self.gu_data[:, :] = data_i[:, :2]
                    self.gu_label[:, 0] = data_i[:, 2]

                    self.gu_models[i].zero_grad()

                    emb, out = self.models[i](self.gu_data)

                    gu_out = self.gu_models[i](emb, out)
                    loss = self.gu_loss_fn(torch.reshape(gu_out[:, 0], (-1, 1)).to(self.device), self.gu_label)

                    loss.backward()
                    self.gu_optimizers[i].step()

                    self.gu_losses[i].append(loss)

class MNIST:
    def __init__(self, data, hyp,hypgen, device):

        self.device = device

        self.X_train, self.y_train, self.X_test, self.y_test = data[0], data[1], data[2], data[3]

        self.datasets = []
        self.models = []
        self.optimizers = []
        self.lr_schedulers = []
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = hyp["lr"]
        self.test_dataset = create_MNIST_dataset(self.X_test, self.y_test)
        self.n_device = hypgen["n_device"]
        self.n_class = hypgen["n_class"]
        self.obs_clss = hypgen["observed_classes"]
        for i in range(self.n_device):
            filt = [True if j in hypgen["base_classes"][i] else False for j in self.y_train]
            datas = create_MNIST_dataset(self.X_train[filt], self.y_train[filt])
            self.datasets.append(datas)
            model_i = MNISTClassifier().to(device)
            model_i.load_state_dict(torch.load(hyp["weight_loc"]+ "basemodel"+str(i)+".pt"))
            self.models.append(model_i)
            opts = optim.SGD(self.models[i].parameters(), lr=self.lr, weight_decay=0.01)
            self.optimizers.append(opts)
            lr_sch = lr_scheduler.ExponentialLR(self.optimizers[i], gamma=0.9, last_epoch=-1)
            self.lr_schedulers.append(lr_sch)

        self.T = hyp["T"]
        self.N = hyp["N"]
        self.cache_size = hyp["cache_size"]
        self.round_numb = hyp["round_num"]
        self.num_epoch = hyp["num_epoch"]
        self.b_size = hyp["b_size"]
        self.losses = [[] for i in range(self.n_device)]
        self.nimgs = [[] for i in range(self.n_device)]
        self.avgs = [[] for i in range(self.n_device)]
        self.prevs = [0 for i in range(self.n_device)]
        self.all_accs = [[] for i in range(self.n_device)]
        self.ood_accs = [[] for i in range(self.n_device)]

        self.data_cams = [cam_MNIST_filter(hypgen["observed_classes"][i], self.X_train,self.y_train) for i in range(self.n_device)]

        self.data = torch.zeros((self.b_size, 1,28,28), dtype=torch.float).to(device)

        self.label = torch.zeros((self.b_size), dtype=int).to(device)

        self.train_datasets = [i for i in self.datasets]
        self.dataloaders = [[] for i in range(self.n_device)]
        self.ood_classes = [[] for i in range(self.n_device)]
        for i in range(self.n_device):
            for j in hypgen["desired_classes"][i]:
                if j not in hypgen["base_classes"][i]:
                    self.ood_classes[i].append(j)

    def img_generate(self):

        self.imgs = [random.sample(self.data_cams[i],self.cache_size) for i in range(self.n_device)]

    def util_calculate(self):

        utils = [0 for j in range(self.n_device)]

        for i in range(self.n_device):
            for j in range(self.n_device):
                for rc in self.caches[i]:
                    if rc[1] in self.ood_classes[j]:
                        utils[i] += 1
            utils[i] = utils[i]/self.n_device

        for i in range(self.n_device):
            self.prevs[i] += utils[i]
            self.nimgs[i].append(self.prevs[i])
            self.avgs[i].append(utils[i])

    def update_trainset(self):
        caches = []
        for i in range(self.n_device):
            caches += self.caches[i]
        cache_X = torch.tensor([x[0] for x in caches])
        cache_y = torch.tensor([x[1] for x in caches])
        cached_dataset = create_MNIST_dataset(cache_X,cache_y)
        for i in range(self.n_device):
            self.train_datasets[i] =  torch.utils.data.ConcatDataset((self.train_datasets[i], cached_dataset))

    def create_dataloaders(self):

        for i in range(self.n_device):
            self.dataloaders[i] = torch.utils.data.DataLoader(self.train_datasets[i], batch_size=self.b_size, shuffle=True,
                                                       drop_last=True,
                                                       worker_init_fn=0)

    def train_model(self, silent=True):

        for n_i in range(self.n_device):
            self.models[n_i].train()

            if not silent:
                pbar = tqdm([i for i in range(self.num_epoch)], total=self.num_epoch)
            else:
                pbar = [i for i in range(self.num_epoch)]

            for epoch in pbar:
                for x, y in self.dataloaders[n_i]:
                    self.data[:] = x.reshape(self.data.shape)
                    self.label[:] = y

                    self.models[n_i].zero_grad()

                    emb, out = self.models[n_i](self.data)
                    loss = self.loss_fn(out, self.label)
                    loss.backward()

                    self.optimizers[n_i].step()

                if not silent:
                    mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                    s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
                    pbar.set_description(s)

    def test_function(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1)
        data = torch.zeros((1, 1, 28, 28), dtype=torch.float).to(self.device)
        label = torch.zeros((1), dtype=int).to(self.device)
        confusion_matrices = [np.zeros((10, 10), dtype=int) for i in range(self.n_device)]

        for n_i in range(self.n_device):
            self.models[n_i].eval()

            with torch.no_grad():
                for x, y in test_loader:

                    data[:] = x.reshape(data.shape)
                    label[:] = y

                    emb, out = self.models[n_i](data)

                    _, pred = torch.max(out.data, 1)

                    for i in range(10):
                        filt_i = (label == i)

                        pred_i = pred[filt_i]

                        for j in range(10):
                            filt_j = (pred_i == j)

                            nnum = sum(filt_j)

                            confusion_matrices[n_i][i, j] += nnum


        self.confusion_matrices = confusion_matrices

    def acc_calc(self):

        for i in range(self.n_device):
            all_acc = 0
            ood_acc = 0
            for j in range(self.n_class):
                all_acc += self.confusion_matrices[i][j, j] / (np.sum(self.confusion_matrices[i], axis=1)[j])
                if j in self.ood_classes[i]:
                    ood_acc += self.confusion_matrices[i][j, j] / (np.sum(self.confusion_matrices[i], axis=1)[j])

            all_acc /= self.n_class
            ood_acc /= len(self.ood_classes[i])

            self.all_accs[i].append(all_acc)
            self.ood_accs[i].append(ood_acc)

    def save_sim_data(self, name, out_loc):
        self.nimg = [0 for i in range(self.round_numb)]
        self.all_acc = [0 for i in range(self.round_numb)]
        self.ood_acc = [0 for i in range(self.round_numb)]

        for i in range(self.n_device):
            for j in range(self.round_numb):
                self.nimg[j] = self.nimgs[i][j]/self.n_device
                self.all_acc[j] = self.all_accs[i][j]/self.n_device
                self.ood_acc[j] = self.ood_accs[i][j] / self.n_device

        save_list(self.nimg, out_loc, name + "_nimg.npy")
        save_list(self.all_acc, out_loc, name + "_all_acc.npy")
        save_list(self.ood_acc, out_loc, name + "_ood_acc.npy")

class MNISTRandom(MNIST):
    def __init__(self, data, hyp, hypgen,device):
        super().__init__(data, hyp,hypgen, device)

    def cache_generate(self):
        self.caches = []

        for i in range(self.n_device):
            self.caches.append(random.sample(self.imgs[i], self.T))

class MNISTOracle(MNIST):
    def __init__(self, data, hyp,hypgen, device):
        super().__init__(data, hyp,hypgen, device)

    def cache_generate(self):
        self.caches = []
        util_class = [0 for i in range(self.n_class)]
        seen_class = [0 for i in range(self.n_class)]
        for i in range(self.n_device):
            for j in self.ood_classes[i]:
                util_class[j] += 1/self.n_device
            for j in self.obs_clss[i]:
                seen_class[j] += 1

        for i in range(self.n_device):
            cache_ratio = [util_class[o]/seen_class[o] for o in self.obs_clss[i]]
            cache_ratio = [self.T*j/sum(cache_ratio) for j in cache_ratio]
            cache_numb = saferound(cache_ratio,places=0)
            cache_numb = [int(c) for c in cache_numb]
            cc = []
            for j in range(len(self.obs_clss[i])):
                caches_clss = [c for c in self.imgs[i] if c[1] == self.obs_clss[i][j]]
                cc = cc + caches_clss[:cache_numb[j]]
            self.caches.append(cc)

class MNISTSoftmax(MNIST):
    def __init__(self, data, hyp, hypgen, device):
        super().__init__(data, hyp, hypgen, device)

        self.lsofmax = nn.LogSoftmax(dim=1)

    def cache_generate(self):
        self.caches = []

        for i in range(self.n_device):
            X_cam = torch.tensor([j[0] for j in self.imgs[i]]).to(self.device)
            X_cam = X_cam.reshape((-1,1,28,28)).float()
            self.models[i].eval()
            with torch.no_grad():
                _, out = self.models[i](X_cam)
            lsoft_out = self.lsofmax(out)
            lsof_maxs, _ = torch.max(lsoft_out, 1)
            _, ood_ind = torch.topk(-lsof_maxs, self.T)
            self.caches.append([self.imgs[i][j] for j in list(ood_ind)])

class MNISTEntropy(MNIST):

    def __init__(self, data, hyp,hypgen, device):
        super().__init__(data, hyp,hypgen, device)
        self.sofmax = nn.Softmax(dim=1)

    def cache_generate(self):
        self.caches = []

        for i in range(self.n_device):
            X_cam = torch.tensor([j[0] for j in self.imgs[i]]).to(self.device)
            X_cam = X_cam.reshape((-1,1,28,28)).float()
            self.models[i].eval()
            with torch.no_grad():
                _, out = self.models[i](X_cam)
            soft_out = self.sofmax(out)
            log_outs = torch.log(soft_out)
            entr = torch.sum(-soft_out * log_outs, 1)
            _, ood_ind = torch.topk(entr, self.T)
            self.caches.append([self.imgs[i][j] for j in list(ood_ind)])

class MNISTScodselect(MNIST):
    def __init__(self, data, hyp,hypgen, device):
        super().__init__(data, hyp,hypgen, device)
        self.dist_fam = CategoricalLogit().to(device)
        self.kwargs = {'num_samples': 64, 'num_eigs': 10, 'device': device, 'sketch_type': 'srft'}

    def prepro(self):
        self.scods = [SCOD(self.models[i],self.dist_fam,self.kwargs) for i in range(self.n_device)]
        for i in range(self.n_device):

            self.scods[i].process_dataset(self.train_datasets[i])

    def cache_generate(self):
        self.caches = []

        for i in range(self.n_device):

            X_cam = torch.tensor([j[0] for j in self.imgs[i]]).to(self.device)
            X_cam = X_cam.reshape((-1, 1, 28, 28)).float()
            self.models[i].eval()
            _, _, unc = self.scods[i](X_cam)
            _, ood_ind = torch.topk(unc, self.T, 0)
            self.caches.append([self.imgs[i][j] for j in list(ood_ind)])

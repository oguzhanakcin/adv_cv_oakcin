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

def load_MNIST_dataset(save_dir,n_samples):

    mnist_trainset = datasets.MNIST(root=save_dir, train=True, download=True)
    mnist_testset = datasets.MNIST(root=save_dir, train=False, download=True)

    X_train = mnist_trainset.data
    y_train = mnist_trainset.targets
    X_trains = X_train[:n_samples*10]
    y_trains = y_train[:n_samples*10]

    for i in range(10):
        X_trains[i*n_samples:(i+1)*n_samples] = X_train[y_train== i][:n_samples]
        y_trains[i*n_samples:(i+1)*n_samples] = y_train[y_train == i][:n_samples]

    X_test = mnist_testset.data
    y_test = mnist_testset.targets
    X_tests = X_train[:n_samples]
    y_tests = y_train[:n_samples]

    for i in range(10):
        X_tests[i * int(n_samples/10):(i + 1) * int(n_samples/10)] = X_test[y_test == i][:int(n_samples/10)]
        y_tests[i * int(n_samples/10):(i + 1) * int(n_samples/10)] = y_test[y_test == i][:int(n_samples/10)]

    return X_trains, y_trains, X_tests, y_tests

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

class GUSampler(Synthetic):
    def __init__(self, data, hyp,hypgen, device):
        super().__init__(data,hyp,hypgen,device)

        self.gu_models = []
        self.gu_optimizers = []
        self.gu_lr_schedulers = []
        self.gu_lr = hyp["gu_lr"]
        self.gu_loss_fn = nn.BCELoss()

        for i in range(self.n_device):
            gu_model = GU().to(device)
            gu_model.apply(init_weights)
            self.gu_models.append(gu_model)
            optimizer = optim.SGD(self.gu_models[i].parameters(), lr=self.gu_lr, weight_decay=0.01)
            self.gu_optimizers.append(optimizer)
            gu_lr_scheduler = lr_scheduler.ExponentialLR(self.gu_optimizers[i], gamma=0.9, last_epoch=-1)
            self.gu_lr_schedulers.append(gu_lr_scheduler)

        self.sofmax = nn.Softmax(dim=1)

        self.gu_losses = [[] for i in range(self.n_device)]
        self.gu_data = torch.zeros((self.T, 2), dtype=torch.float).to(device)
        self.gu_label = torch.zeros((self.T, 1), dtype=torch.float).to(device)

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

            caches_i = torch.tensor(self.caches[i]).to(self.device)
            c_i = caches_i[:,:2]
            l_i = caches_i[:,2]
            lk = []
            for k in l_i:
                lk.append(0)
                for j in range(self.n_device):
                    if int(k) in self.ood_classes[j]:
                        lk[-1] = lk[-1] + 1 / self.n_device
            l_k = torch.tensor(lk).to(self.device)

            cl_i = torch.cat((c_i, torch.reshape(l_k, (-1, 1))), 1)

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

class MNISTGUSampler(MNIST):
    def __init__(self, data, hyp,hypgen, device):
        super().__init__(data,hyp,hypgen,device)

        self.gu_models = []
        self.gu_optimizers = []
        self.gu_lr_schedulers = []
        self.gu_lr = hyp["gu_lr"]
        self.gu_loss_fn = nn.BCELoss()

        for i in range(self.n_device):
            gu_model = MNISTGU().to(device)
            gu_model.apply(init_weights)
            self.gu_models.append(gu_model)
            optimizer = optim.SGD(self.gu_models[i].parameters(), lr=self.gu_lr, weight_decay=0.01)
            self.gu_optimizers.append(optimizer)
            gu_lr_scheduler = lr_scheduler.ExponentialLR(self.gu_optimizers[i], gamma=0.9, last_epoch=-1)
            self.gu_lr_schedulers.append(gu_lr_scheduler)

        self.sofmax = nn.Softmax(dim=1)

        self.gu_losses = [[] for i in range(self.n_device)]
        self.gu_data = torch.zeros((self.T, 1,28,28), dtype=torch.float).to(device)
        self.gu_label = torch.zeros((self.T), dtype=torch.float).to(device)

    def cache_generate(self):
        self.caches = []

        for i in range(self.n_device):
            X_cam = torch.tensor([j[0] for j in self.imgs[i]]).to(self.device)
            X_cam = X_cam.reshape((-1,1,28,28)).float()
            self.models[i].eval()
            self.gu_models[i].eval()

            with torch.no_grad():
                emb, out = self.models[i](X_cam)
                score = self.gu_models[i](emb,out)

            scores = score.cpu().detach().numpy()

            ind = np.random.choice(len(scores), self.T, p=scores[:,0] / scores.sum(), replace=False)
            self.caches.append([self.imgs[i][j] for j in list(ind)])

    def gu_train(self):

        for i in range(self.n_device):

            self.gu_models[i].train()
            self.models[i].eval()

            cache_X = torch.tensor([x[0] for x in self.caches[i]])
            cache_y = torch.tensor([x[1] for x in self.caches[i]])
            lk = []
            for k in cache_y:
                lk.append(0)
                for j in range(self.n_device):
                    if int(k) in self.ood_classes[j]:
                        lk[-1] = lk[-1] + 1 / self.n_device
            l_k = torch.tensor(lk)

            transform = trfm.Normalize((0.1307,), (0.3081,))
            cached_dataset = MNISTGUDataset(cache_X,l_k,transform)

            dataloader = torch.utils.data.DataLoader(cached_dataset, batch_size=self.T,
                                                                  shuffle=True,
                                                                  drop_last=True,
                                                                  worker_init_fn=0)


            pbar = [i for i in range(20)]

            for epoch in pbar:
                for x,y in dataloader:

                    self.gu_data[:] = x.reshape(self.gu_data.shape)
                    self.gu_label[:] = y

                    self.gu_models[i].zero_grad()

                    emb, out = self.models[i](self.gu_data)

                    gu_out = self.gu_models[i](emb, out)
                    loss = self.gu_loss_fn(gu_out[:, 0], self.gu_label)

                    loss.backward()
                    self.gu_optimizers[i].step()

                    self.gu_losses[i].append(loss)
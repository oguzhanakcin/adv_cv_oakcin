import torch
import torchvision.models as vsmodels
from torch.utils.data import Dataset
import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
import torch.nn as nn
from utils import *

class WpredGU(nn.Module):
    def __init__(self):
        super(WpredGU, self).__init__()
        self.gu_prob = nn.Sequential(
            nn.Linear(516, 10),
            nn.ReLU(True),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self,  emb,logit):
        x = torch.cat((emb,logit), 1)
        out = self.gu_prob(x)
        return out

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class Wpred:
    def __init__(self, data, hyp,hypgen, device,dataset_loc):

        self.device = device

        self.X_train, self.y_train, self.X_test, self.y_test = data[0], data[1], data[2], data[3]

        self.datasets = []
        self.models = []
        self.optimizers = []
        self.lr_schedulers = []
        self.dataset_loc = dataset_loc
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = hyp["lr"]
        self.test_dataset = create_Weather_dataset(self.X_test, self.y_test,dataset_loc)

        self.n_device = hypgen["n_device"]
        self.n_class = hypgen["n_class"]
        self.obs_clss = hypgen["observed_classes"]

        for i in range(self.n_device):

            X_train = [self.X_train[j] for j in range(len(self.y_train)) if
                       self.y_train[j] in hypgen["base_classes"][i]]

            y_train = [self.y_train[j] for j in range(len(self.y_train)) if
                       self.y_train[j] in hypgen["base_classes"][i]]

            datas = create_Weather_dataset(X_train, y_train,dataset_loc)
            self.datasets.append(datas)

            model_i = vsmodels.resnet18().to(device)
            model_i.fc = nn.Linear(in_features=512, out_features=4).to(device)
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

        self.data_cams = [cam_Weather_filter(hypgen["observed_classes"][i], self.X_train,self.y_train) for i in range(self.n_device)]
        
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

        cache_X = [x[0] for x in caches]
        cache_y = [x[1] for x in caches]
        cached_dataset = create_Weather_dataset(cache_X,cache_y,self.dataset_loc)
        for i in range(self.n_device):
            self.train_datasets[i] =  torch.utils.data.ConcatDataset((self.train_datasets[i], cached_dataset))
    
    def create_dataloaders(self):
        for i in range(self.n_device):
            self.dataloaders[i], self.data, self.label = create_Weather_dataloader(self.train_datasets[i], self.b_size, self.device)

    def train_model(self):

        for n_i in range(self.n_device):

            self.models[n_i].train()

            pbar = [i for i in range(self.num_epoch)]

            for epoch in pbar:
                for x, y in self.dataloaders[n_i]:
                    self.data[:] = x.reshape(self.data.shape)
                    self.label[:] = y

                    self.models[n_i].zero_grad()

                    out = self.models[n_i](self.data)
                    loss = self.loss_fn(out, self.label)
                    loss.backward()

                    self.optimizers[n_i].step()

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
            ood_acc /= len(self.ood_classes[i])+1e-9

            self.all_accs[i].append(all_acc)
            self.ood_accs[i].append(ood_acc)
            
    def save_sim_data(self,name,out_loc):
        
        self.nimg = [0 for i in range(self.round_numb)]
        self.all_acc = [0 for i in range(self.round_numb)]
        self.ood_acc = [0 for i in range(self.round_numb)]

        for i in range(self.n_device):
            for j in range(self.round_numb):
                self.nimg[j] += self.nimgs[i][j]/self.n_device
                self.all_acc[j] += self.all_accs[i][j]/self.n_device
                self.ood_acc[j] += self.ood_accs[i][j] / self.n_device

        save_list(self.nimg, out_loc, name + "_nimg.npy")
        save_list(self.all_acc, out_loc, name + "_all_acc.npy")
        save_list(self.ood_acc, out_loc, name + "_ood_acc.npy")

class WpredRandom(Wpred):
    def __init__(self, data, hyp, hypgen, device,dataset_loc):
        super().__init__(data, hyp, hypgen, device,dataset_loc)

    def cache_generate(self):
        self.caches = []

        for i in range(self.n_device):
            self.caches.append(random.sample(self.imgs[i], self.T))

class WpredOracle(Wpred):
    def __init__(self, data, hyp, hypgen, device,dataset_loc):
        super().__init__(data, hyp, hypgen, device,dataset_loc)

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

class WpredSoftmax(Wpred):
    def __init__(self, data, hyp, hypgen, device,dataset_loc):
        super().__init__(data, hyp, hypgen, device,dataset_loc)

        self.lsofmax = nn.Softmax(dim=1)

    def cache_generate(self):
        self.caches = []

        for i in range(self.n_device):

            X = [x[0] for x in self.imgs[i]]
            y = [x[1] for x in self.imgs[i]]
            dataset = create_Weather_dataset(X,y,self.dataset_loc)

            dataloader,data,label = create_Weather_dataloader(dataset, self.cache_size, self.device)
        
            self.models[i].eval()
            with torch.no_grad():
                for x,y in dataloader:
                    data[:] = x.reshape(data.shape)
                    out = self.models[i](data)
            
            lsoft_out = self.lsofmax(out)
            lsof_maxs, _ = torch.max(lsoft_out, 1)
            _, ood_ind = torch.topk(-lsof_maxs, self.T)
            self.caches.append([self.imgs[i][j] for j in list(ood_ind)])

class WpredEntropy(Wpred):

    def __init__(self, data, hyp,hypgen, device,dataset_loc):
        super().__init__(data, hyp,hypgen, device,dataset_loc)
        self.sofmax = nn.Softmax(dim=1)

    def cache_generate(self):
        self.caches = []

        for i in range(self.n_device):

            X = [x[0] for x in self.imgs[i]]
            y = [x[1] for x in self.imgs[i]]
            dataset = create_Weather_dataset(X,y,self.dataset_loc)

            dataloader,data,label = create_Weather_dataloader(dataset, self.cache_size, self.device)
        
            self.models[i].eval()
            with torch.no_grad():
                for x,y in dataloader:
                    data[:] = x.reshape(data.shape)
                    out = self.models[i](data)
            
            soft_out = self.sofmax(out)
            log_outs = torch.log(soft_out)
            entr = torch.sum(-soft_out * log_outs, 1)
            _, ood_ind = torch.topk(entr, self.T)
            self.caches.append([self.imgs[i][j] for j in list(ood_ind)])

class WpredGUSampler(Wpred):
    def __init__(self, data, hyp,hypgen, device,dataset_loc):
        super().__init__(data, hyp,hypgen, device,dataset_loc)
        
        self.gu_models = []
        self.gu_optimizers = []
        self.gu_lr_schedulers = []
        self.gu_lr = hyp["gu_lr"]
        self.gu_loss_fn = nn.BCELoss()

        for i in range(self.n_device):
            gu_model = WpredGU().to(device)
            gu_model.apply(init_weights)
            self.gu_models.append(gu_model)
            optimizer = optim.SGD(self.gu_models[i].parameters(), lr=self.gu_lr, weight_decay=0.01)
            self.gu_optimizers.append(optimizer)
            self.models[i].avgpool.register_forward_hook(get_activation('avgpool'))
            gu_lr_scheduler = lr_scheduler.ExponentialLR(self.gu_optimizers[i], gamma=0.9, last_epoch=-1)
            self.gu_lr_schedulers.append(gu_lr_scheduler)

        self.gu_losses = [[] for i in range(self.n_device)]
    
    def cache_generate(self):

        self.caches = []

        for i in range(self.n_device):

            X = [x[0] for x in self.imgs[i]]
            y = [x[1] for x in self.imgs[i]]
            dataset = create_Weather_dataset(X,y,self.dataset_loc)

            dataloader,data,label = create_Weather_dataloader(dataset, self.cache_size, self.device)
        
            self.models[i].eval()
            with torch.no_grad():
                for x,y in dataloader:
                    data[:] = x.reshape(data.shape)
                    out = self.models[i](data)
                    emb = activation["avgpool"]
                    score = self.gu_models[i](emb.reshape((-1,512)),out)
            
            scores = score.cpu().detach().numpy()

            ind = np.random.choice(len(scores), self.T, p=scores[:,0] / scores.sum(), replace=False)
            self.caches.append([self.imgs[i][j] for j in list(ind)])

    def gu_train(self):
        
        for i in range(self.n_device):

            self.gu_models[i].train()
            self.models[i].eval()

            X = [x[0] for x in self.caches[i]]
            y = [x[1] for x in self.caches[i]]
            lk = []
            for k in y:
                lk.append(0)
                for j in range(self.n_device):
                    if int(k) in self.ood_classes[j]:
                        lk[-1] = lk[-1] + 1 / self.n_device
            l_k = torch.tensor(lk)

            dataset = create_Weather_dataset(X,l_k,self.dataset_loc)

            dataloader,data,label = create_Weather_dataloader(dataset, self.T, self.device)


            pbar = [i for i in range(20)]

            for epoch in pbar:
                for x,y in dataloader:

                    data[:] = x.reshape(data.shape)

                    self.gu_models[i].zero_grad()

                    out = self.models[i](data)
                    emb = activation["avgpool"]

                    gu_out = self.gu_models[i](emb.reshape((-1,512)), out)
                    loss = self.gu_loss_fn(gu_out[:, 0], y.to(self.device))

                    loss.backward()
                    self.gu_optimizers[i].step()

                    self.gu_losses[i].append(loss) 




            
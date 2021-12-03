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

class Synthetic:

    def __init__(self, data, hyp, device):

        self.device = device

        self.X_train, self.X_test, self.y_train, self.y_test = data[0], data[1], data[2], data[3]

        self.datasets = []
        self.models = []
        self.optimizers = []
        self.lr_schedulers = []
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = hyp["lr"]
        self.test_dataset = create_dataset(self.X_test, self.y_test)
        self.n_device = hyp["n_device"]
        self.n_class = hyp["n_class"]
        self.obs_clss = hyp["observed_classes"]
        for i in range(self.n_device):
            filt = [True if j in hyp["base_classes"][i] else False for j in self.y_train]
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

        self.data_cams = [cam_filter(hyp["observed_classes"][i], self.X_train,self.y_train) for i in range(self.n_device)]

        self.data = torch.zeros((self.b_size, 2), dtype=torch.float).to(device)

        self.label = torch.zeros((self.b_size), dtype=int).to(device)

        self.train_datasets = [i.to(device) for i in self.datasets]
        self.dataloaders = [[] for i in range(self.n_device)]
        self.ood_classes = [[] for i in range(self.n_device)]
        for i in range(self.n_device):
            for j in hyp["desired_classes"][i]:
                if j not in hyp["base_classes"][i]:
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
    def __init__(self, data, hyp, device):
        super().__init__(data, hyp,device)

    def cache_generate(self):
        self.caches = []

        for i in range(self.n_device):
            self.caches.append(random.sample(self.imgs[i], self.T))

class Oracle(Synthetic):
    def __init__(self, data, hyp,device):
        super().__init__(data, hyp,device)

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

    def __init__(self, data, hyp, device):
        super().__init__(data, hyp, device)

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

    def __init__(self, data, hyp, device):
        super().__init__(data, hyp, device)
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
    def __init__(self, data, hyp, device):
        super().__init__(data, hyp, device)
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

class GUSampler(Synthetic):
    def __init__(self, data, hyp,device):
        super().__init__(data,hyp,device)

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

    def gu_train(self,i):

        for i in range(self.n_device):

            self.gu_models[i].train()
            self.models[i].eval()

            c_i = torch.tensor(self.caches[i])[:, :2].to(self.device)
            l_i = torch.tensor(self.caches[i])[:, 2].to(self.device)
            lt_i = [(j, l_i.int().tolist()[j]) for j in range(len(l_i))]

            tlos = 0

            for j in range(self.n_device):
                _,o = self.models[j](c_i)
                outs = self.sofmax(o)
                tilos = 1 - torch.tensor([outs[lt_i[k]].tolist() for k in range(len(lt_i))])

                tlos += tilos.to(self.device)/self.n_device

            cl_i = torch.cat((c_i, torch.reshape(tlos, (-1, 1))), 1)

            dataloader = torch.utils.data.DataLoader(cl_i, batch_size=self.T, shuffle=True, drop_last=True,
                                                  worker_init_fn=0)
            pbar = [i for i in range(self.num_epoch)]

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
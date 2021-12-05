import argparse
import yaml
import sys
from torch.utils.data import Dataset
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as vsmodels
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from utils import *
from models import *

def simulate(data,device,hyp,hypgen,out_loc,sim_type,dataset_loc):

    if sim_type == "random":
        sim = WpredRandom(data,hyp, hypgen,device,dataset_loc)
    elif sim_type == "oracle":
        sim = WpredOracle(data,hyp, hypgen,device,dataset_loc)
    elif sim_type == "softmax":
        sim = WpredSoftmax(data, hyp, hypgen, device,dataset_loc)
    elif sim_type == "entropy":
        sim = WpredEntropy(data, hyp, hypgen, device,dataset_loc)
    elif sim_type == "scod":
        sim = WpredScodselect(data, hyp, hypgen, device,dataset_loc)
    elif sim_type == "gu":
        sim = WpredGUSampler(data,hyp,hypgen,device,dataset_loc)

    bar = tqdm([i for i in range(sim.round_numb)], total=sim.round_numb)

    for i in bar:

        if (sim_type == "scod" ):
            sim.prepro()

        random.seed(i)

        sim.img_generate()

        sim.cache_generate()

        sim.util_calculate()

        sim.update_trainset()

        sim.create_dataloaders()

        sim.train_model()

        if sim_type == "gu":
            sim.gu_train()

        sim.test_function()

        sim.acc_calc()

        mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        avgs = 0
        all_accs = 0
        for i in range(sim.n_device):
            avgs += sim.avgs[i][-1] / sim.n_device
            all_accs += sim.all_accs[i][-1] / sim.n_device
        s = ("%10s" * 1 + "%10.4g" * 2) % (mem, avgs,all_accs)
        bar.set_description(s)

    sim.save_sim_data(sim_type,out_loc)

def train_base(data,device,hyp,hypgen,dataset_loc):

    n_device = hypgen["n_device"]
    base_clss = hypgen["base_classes"]
    n_clss = hypgen["n_class"]
    X_train,y_train, X_test, y_test = data[0], data[1] , data[2] , data[3]

    for n_i in range(n_device):


        base_model = vsmodels.resnet18(pretrained=True).to(device)
        base_model.fc = nn.Linear(in_features=512,out_features=4).to(device)

        X_trains = [X_train[i] for i in range(len(y_train)) if y_train[i] in base_clss[n_i]]
        y_trains = [y_train[i] for i in range(len(y_train)) if y_train[i] in base_clss[n_i]]

        dataset = create_Weather_dataset(X_trains,y_trains,dataset_loc)
        test_dataset = create_Weather_dataset(X_test,y_test,dataset_loc)

        loss_fn = nn.CrossEntropyLoss()
        lr = hyp["lr"]
        optimizer = optim.SGD(base_model.parameters(), lr=lr,weight_decay=0.01)
        lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.9,last_epoch=-1)

        num_epoch = hyp["num_epoch"]
        b_size = hyp["b_size"]

        losses = []

        dataloader, data, label = create_Weather_dataloader(dataset , b_size, device)

        base_model.train()
        pbar = tqdm([i for i in range(num_epoch)], total=num_epoch)

        for epoch in pbar:
            for x, y in dataloader:
                data[:] = x.reshape(data.shape)
                label[:] = y

                base_model.zero_grad()

                out = base_model(data)
                loss = loss_fn(out, label)
                losses.append(loss)

                loss.backward()

                optimizer.step()

            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
            pbar.set_description(s)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        data = torch.zeros((1, 3, 224, 224), dtype=torch.float).to(device)
        label = torch.zeros((1), dtype=int).to(device)

        confusion_matrix = np.zeros((n_clss, n_clss), dtype=int)
        base_model.eval()
        with torch.no_grad():
            for x, y in test_loader:

                data[:] = x.reshape(data.shape)
                label[:] = y

                out = base_model(data)

                _, pred = torch.max(out.data, 1)

                for i in range(n_clss):
                    filt_i = (label == i)
                    pred_i = pred[filt_i]
                    for j in range(n_clss):
                        filt_j = (pred_i == j)
                        nnum = sum(filt_j)
                        confusion_matrix[i, j] += nnum

        all_accs,ood_accs = acc_calc(confusion_matrix,n_clss,base_clss[n_i])

        print("Accuracy of All :",'{0:.3g}'.format(all_accs))
        print("Accuracy of OoD :",'{0:.3g}'.format(ood_accs))

        isExist = os.path.exists(hyp["save_loc"])

        if not isExist:
            os.makedirs(hyp["save_loc"])

        
        torch.save(base_model.state_dict(), hyp["save_loc"]+"/basemodel"+str(n_i)+".pt")

def run_sim(hyp,opt,device):

    if opt.create_label:
        print("Creating Label")
        create_labels_Wpred(opt.dataset_loc,opt.dataset_loc,opt.train_ratio)

    with open(opt.dataset_loc + "/train.yaml") as f:
        train_dict = yaml.load(f,Loader=yaml.SafeLoader)
    train_imgs_locs = list(train_dict.keys())
    train_labels = [train_dict[i] for i in train_imgs_locs]

    with open(opt.dataset_loc + "/test.yaml") as f:
        test_dict = yaml.load(f, Loader=yaml.SafeLoader)

    test_imgs_locs = list(test_dict.keys())
    test_labels = [test_dict[i] for i in test_imgs_locs]

    data = [train_imgs_locs,train_labels,test_imgs_locs,test_labels]


    if opt.train_base:
        print("Training base model")
        train_base(data,device,hyp["base_model"],hyp["general"],opt.dataset_loc)

    if opt.sim_random:
        print("Simulating random model")
        simulate(data,device,hyp["random_model"],hyp["general"],opt.sim_result_loc,"random",opt.dataset_loc)

    if opt.sim_oracle:
        print("Simulating oracle model")
        simulate(data,device,hyp["oracle_model"],hyp["general"],opt.sim_result_loc,"oracle",opt.dataset_loc)

    if opt.sim_soft:
        print("Simulating softmax model")
        simulate(data,device,hyp["soft_model"],hyp["general"],opt.sim_result_loc,"softmax",opt.dataset_loc)

    if opt.sim_entropy:
        print("Simulating Entropy model")
        simulate(data, device, hyp["entropy_model"],hyp["general"], opt.sim_result_loc, "entropy",opt.dataset_loc)

    if opt.sim_scod:
        print("Simulating Scod model")
        simulate(data, device, hyp["scod_model"],hyp["general"], opt.sim_result_loc, "scod",opt.dataset_loc)

    if opt.sim_gu:
        print("Simulating Gu model")
        simulate(data, device, hyp["gu_model"],hyp["general"], opt.sim_result_loc, "gu",opt.dataset_loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-label", action= "store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dataset-loc", type=str,default="./data")
    parser.add_argument("--hyp-loc",type=str,default="hyp.yaml")
    parser.add_argument("--train-base", action="store_true")
    parser.add_argument("--sim-result-loc", type=str, default="sim_data")
    parser.add_argument("--sim-random",action="store_true")
    parser.add_argument("--sim-oracle",action="store_true")
    parser.add_argument("--sim-soft",action="store_true")
    parser.add_argument("--sim-entropy",action="store_true")
    parser.add_argument("--sim-scod",action="store_true")
    parser.add_argument("--sim-gu", action="store_true")

    opt = parser.parse_args()

    with open(opt.hyp_loc) as f:
        hyp = yaml.load(f,Loader=yaml.SafeLoader)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print('Using torch %s %s' % (
    torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

    run_sim(hyp,opt,device)
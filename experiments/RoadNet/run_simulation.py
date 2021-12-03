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

def ood_performance(data,device,hyp,hypgen,out_loc,sim_type,dataset_loc):

    if sim_type == "Softmax":
        sim = RoadNetSoftmax(data,hyp,hypgen,device,dataset_loc)
    elif sim_type == "Entropy":
        sim = RoadNetEntropy(data,hyp,hypgen,device,dataset_loc)
    elif sim_type == "Scod":
        sim = RoadNetScodselect(data,hyp,hypgen,device,dataset_loc)

    if sim_type == "Scod":
        print("Preprocessing Traning Set")
        sim.prepro()
        print("Preprocessing Traning Set is Done")

    sim.ood_scores()
    sim.save_ood_scores(sim_type,out_loc)

def train_base(data,device,hyp,hypgen,dataset_loc):

    n_device = hypgen["n_device"]
    base_clss = hypgen["base_classes"]
    n_clss = hypgen["n_class"]
    model_type = hypgen["model_size"]
    X_train,y_train, X_test, y_test = data[0], data[1] , data[2] , data[3]

    for n_i in range(n_device):
        if model_type == "small":
            base_model = vsmodels.resnet18(pretrained=True).to(device)
            base_model.fc = nn.Linear(in_features=512,out_features=4).to(device)
        elif model_type== "medium":
            base_model = vsmodels.resnet50(pretrained=True).to(device)
            base_model.fc = nn.Linear(in_features=2048, out_features=4).to(device)
        else :
            base_model = vsmodels.resnet101(pretrained=True).to(device)
            base_model.fc = nn.Linear(in_features=2048, out_features=4).to(device)

        X_train = [X_train[i] for i in range(len(y_train)) if y_train[i] in base_clss[n_i]]
        y_train = [y_train[i] for i in range(len(y_train)) if y_train[i] in base_clss[n_i]]

        dataset = create_RoadNet_dataset(X_train,y_train,dataset_loc)
        test_dataset = create_RoadNet_dataset(X_test,y_test,dataset_loc)

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

            isExist = os.path.exists(hyp["save_loc"])

            if not isExist:
                os.makedirs(hyp["save_loc"])

            if model_type == "small":
                torch.save(base_model.state_dict(), hyp["save_loc"] + "/smallbasemodel" + str(n_i) + ".pt")
            else:
                torch.save(base_model.state_dict(), hyp["save_loc"] + "/largebasemodel" + str(n_i) + ".pt")

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

        if model_type == "small":
            torch.save(base_model.state_dict(), hyp["save_loc"]+"/smallbasemodel"+str(n_i)+".pt")
        else:
            torch.save(base_model.state_dict(), hyp["save_loc"] + "/largebasemodel" + str(n_i) + ".pt")

def run_sim(hyp,opt,device):


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

    if opt.sim_ood_score:
        print(opt.sim_type + " OoD Scores")
        ood_performance(data, device, hyp["scod_model"], hyp["general"], opt.sim_result_loc, opt.sim_type,opt.dataset_loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dataset-loc", type=str,default="./../../../drive/MyDrive/RoadNet")
    parser.add_argument("--hyp-loc",type=str,default="hyp.yaml")
    parser.add_argument("--train-base", action="store_true")
    parser.add_argument("--sim-result-loc", type=str, default="sim_data")
    parser.add_argument("--sim-ood-score", action="store_true")
    parser.add_argument("--sim-type", type=str, default="Scod")

    opt = parser.parse_args()


    with open(opt.hyp_loc) as f:
        hyp = yaml.load(f,Loader=yaml.SafeLoader)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print('Using torch %s %s' % (
    torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

    run_sim(hyp,opt,device)
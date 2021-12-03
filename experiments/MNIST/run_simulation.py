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


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from utils import *
from models import *

def ood_performance(data,device,hyp,hypgen,out_loc,sim_type):

    if sim_type == "Softmax":
        sim = MNISTSoftmax(data,hyp,hypgen,device)
    elif sim_type == "Entropy":
        sim = MNISTEntropy(data,hyp,hypgen,device)
    elif sim_type == "Scod":
        sim = MNISTScodselect(data,hyp,hypgen,device)

    if sim_type == "Scod":
        print("Preprocessing Traning Set")
        sim.prepro()
        print("Preprocessing Traning Set is Done")

    sim.ood_scores()
    sim.save_ood_scores(sim_type,out_loc)

def train_base(data,device,hyp,hypgen):

    n_device = hypgen["n_device"]
    base_clss = hypgen["base_classes"]
    n_clss = hypgen["n_class"]
    X_train,y_train, X_test, y_test = data[0], data[1] , data[2] , data[3]

    for n_i in range(n_device):

        base_model = MNISTClassifier().to(device)
        base_model.apply(init_weights)

        filt = [True if j in base_clss[n_i] else False for j in y_train]
        dataset = create_MNIST_dataset(X_train[filt],y_train[filt])
        test_dataset = create_MNIST_dataset(X_test,y_test)

        loss_fn = nn.CrossEntropyLoss()
        lr = hyp["lr"]
        optimizer = optim.SGD(base_model.parameters(), lr=lr,weight_decay=0.01)
        lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.9,last_epoch=-1)

        num_epoch = hyp["num_epoch"]
        b_size = hyp["b_size"]

        losses = []

        dataloader, data, label = create_MNIST_dataloader(dataset , b_size, device)

        base_model.train()
        pbar = tqdm([i for i in range(num_epoch)], total=num_epoch)

        for epoch in pbar:
            for x, y in dataloader:
                data[:] = x.reshape(data.shape)
                label[:] = y

                base_model.zero_grad()

                emb, out = base_model(data)
                loss = loss_fn(out, label)
                losses.append(loss)

                loss.backward()

                optimizer.step()

            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
            pbar.set_description(s)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        data = torch.zeros((1, 1, 28, 28), dtype=torch.float).to(device)
        label = torch.zeros((1), dtype=int).to(device)

        confusion_matrix = np.zeros((n_clss, n_clss), dtype=int)
        base_model.eval()
        with torch.no_grad():
            for x, y in test_loader:

                data[:] = x.reshape(data.shape)
                label[:] = y

                emb, out = base_model(data)

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

    X_train,y_train, X_test,y_test = load_MNIST_dataset(opt.dataset_loc)
    data = [X_train,y_train, X_test,y_test]

    if opt.train_base:
        print("Training base model")
        train_base(data,device,hyp["base_model"],hyp["general"])

    if opt.sim_ood_score:
        print(opt.sim_type + " OoD Scores")
        ood_performance(data, device, hyp["scod_model"], hyp["general"], opt.sim_result_loc, opt.sim_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-loc", type=str,default="./data")
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
import argparse
import yaml
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import *
from models import *

def sim_rounds(data,device,hyp,out_loc,sim_type):

    if sim_type == "random":
        sim = Random(data,hyp, device)
    elif sim_type == "oracle":
        sim = Oracle(data,hyp, device)
    elif sim_type == "softmax":
        sim = Softmax(data, hyp, device)
    elif sim_type == "entropy":
        sim = Entropy(data, hyp, device)
    elif sim_type == "scod":
        sim = Scodselect(data, hyp, device)
    elif sim_type == "gu":
        sim = GUSampler(data,hyp, device)

    bar = tqdm([i for i in range(sim.round_numb)], total=sim.round_numb)

    for i in bar:

        if (sim_type == "scod"):
            sim.prepro()

        random.seed(i)

        sim.img_generate()

        sim.cache_generate()

        sim.util_calculate()

        sim.train_model()

        if sim_type =="gu":
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


def train_base(data,device,hyp,hypgen):

    n_device = hypgen["n_device"]
    base_clss = hypgen["base_classes"]
    n_clss = hypgen["n_class"]
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]

    for n_i in range(n_device):

        base_model = Classifier().to(device)
        base_model.apply(init_weights)

        filt = [True if j in base_clss[n_i] else False for j in y_train]

        dataset = create_dataset(X_train[filt],y_train[filt])

        test_dataset = create_dataset(X_test,y_test)

        loss_fn = nn.CrossEntropyLoss()
        lr = hyp["lr"]
        optimizer = optim.SGD(base_model.parameters(), lr=lr,weight_decay=0.01)
        lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.9,last_epoch=-1)

        num_epoch = hyp["num_epoch"]
        b_size = hyp["b_size"]

        losses = []

        dataloader, data, label = create_dataloader(dataset , b_size, device)

        base_model.train()

        pbar = tqdm([i for i in range(num_epoch)], total=num_epoch)

        for epoch in pbar:
            for data_i in dataloader:
                data[:, :] = data_i[:, :2]
                label[:] = data_i[:, 2]

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
        data = torch.zeros((1, 2), dtype=torch.float).to(device)
        label = torch.zeros((1), dtype=int).to(device)
        confusion_matrix = np.zeros((n_clss, n_clss), dtype=int)
        base_model.eval()
        with torch.no_grad():
            for test_data in test_loader:

                data[:, :] = test_data[:, :2]
                label[:] = test_data[:, 2]

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

    if opt.create_dataset:
        print("Creating Dataset")
        create_synt_dataset(opt.dataset_loc,opt.num_class,opt.num_samples,opt.train_ratio)

    X_train, X_test, y_train, y_test = load_synt_dataset(opt.dataset_loc,False)
    data = [X_train, X_test, y_train, y_test]

    if opt.train_base:
        print("Training base model")
        train_base(data,device,hyp["base_model"],hyp["general"])

    if opt.sim_rounds:
        print(opt.sim_type +" OoD Scores")
        sim_rounds(data,device,hyp["general"],opt.sim_result_loc,opt.sim_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-dataset", action="store_true")
    parser.add_argument("--dataset-loc", type=str,default="./data")
    parser.add_argument("--num-samples",type=int, default=2000)
    parser.add_argument("--num-class", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hyp-loc",type=str,default="hyp.yaml")
    parser.add_argument("--train-base", action="store_true")
    parser.add_argument("--sim-result-loc", type=str, default="sim_data")
    parser.add_argument("--sim-rounds", action="store_true")
    parser.add_argument("--sim-type", type= str, default = "gu")

    opt = parser.parse_args()
    opt.sim_rounds = True
    opt.sim_type = "random"

    with open(opt.hyp_loc) as f:
        hyp = yaml.load(f,Loader=yaml.SafeLoader)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print('Using torch %s %s' % (
    torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

    run_sim(hyp,opt,device)
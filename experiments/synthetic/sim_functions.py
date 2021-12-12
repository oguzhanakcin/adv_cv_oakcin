import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from utils import *

# Function to train the base models for the simulations
def train_base(data,device,hyp,hypgen):

    n_device = hypgen["n_device"]
    base_clss = hypgen["base_classes"]
    n_clss = hypgen["n_class"]
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]

    for i in range(n_device):

        base_model = Classifier().to(device)
        base_model.apply(init_weights)

        filt = [True if j in base_clss[i] else False for j in y_train]

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

        losses = train_model(base_model,losses,loss_fn,optimizer,num_epoch,data,label,dataloader,False)
        confusion_matrix = test_function(base_model,test_dataset,device)

        all_accs,ood_accs = acc_calc(confusion_matrix,n_clss,base_clss[i])

        print("Accuracy of All :",'{0:.3g}'.format(all_accs))
        print("Accuracy of OoD :",'{0:.3g}'.format(ood_accs))

        isExist = os.path.exists(hyp["save_loc"])

        if not isExist:
            os.makedirs(hyp["save_loc"])

        torch.save(base_model.state_dict(), hyp["save_loc"]+"/basemodel"+str(i)+".pt")

# Function to simulate the dataset simulation and rounds
def simulate(data,device,hyp,hypgen,out_loc,sim_type):

    # Based on the simulation tyep corresponding simulation is called
    if sim_type == "random":
        sim = Random(data,hyp, hypgen,device)
    elif sim_type == "oracle":
        sim = Oracle(data,hyp, hypgen,device)
    elif sim_type == "softmax":
        sim = Softmax(data, hyp, hypgen, device)
    elif sim_type == "entropy":
        sim = Entropy(data, hyp, hypgen, device)
    elif sim_type == "gu":
        sim = GUSampler(data,hyp, hypgen, device)

    bar = tqdm([i for i in range(sim.round_numb)], total=sim.round_numb)

    for i in bar:

        random.seed(i)

        sim.img_generate()

        sim.cache_generate()

        sim.util_calculate()

        sim.update_trainset()

        sim.create_dataloaders()

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

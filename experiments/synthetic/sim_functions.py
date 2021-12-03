import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from utils import *
from SCOD_codes.nn_ood.distributions import CategoricalLogit
from SCOD_codes.nn_ood.posteriors.scod import SCOD


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

def simulate(data,device,hyp,hypgen,out_loc,sim_type):

    if sim_type == "random":
        sim = Random(data,hyp, hypgen,device)
    elif sim_type == "oracle":
        sim = Oracle(data,hyp, hypgen,device)
    elif sim_type == "softmax":
        sim = Softmax(data, hyp, hypgen, device)
    elif sim_type == "entropy":
        sim = Entropy(data, hyp, hypgen, device)
    elif sim_type == "scod":
        sim = Scodselect(data, hyp, hypgen, device)
    elif sim_type == "gu":
        sim = GUSampler(data,hyp, hypgen, device)

    bar = tqdm([i for i in range(sim.round_numb)], total=sim.round_numb)

    for i in bar:

        if (sim_type == "scod" or sim_type == "scod_gu"):
            sim.prepro()
            if sim_type == "scod_gu":
                sim.gu_unc_scores()

        random.seed(i)

        sim.img_generate()

        sim.cache_generate()

        sim.util_calculate()

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

def ood_num_effect(data,device,hyp):
  X_train,X_test,y_train,y_test = data[0],data[1],data[2],data[3]
  numb_of_ood = hyp["num_of_ood"]

  classifiers = [Classifier().to(device)]
  classifiers[0].apply(init_weights)

  for i in range(len(numb_of_ood)-1):
    classifiers.append(copy.deepcopy(classifiers[0]))

  loss_fn = nn.CrossEntropyLoss()

  lr = hyp["lr"]
  optimizers = [optim.SGD(classifiers[i].parameters(), lr=lr,weight_decay=0.01) for i in range(len(classifiers))]

  lr_schs = [lr_scheduler.ExponentialLR(optimizers[i],gamma=0.9,last_epoch=-1) for i in range(len(classifiers))]

  num_epoch = hyp["num_epoch"]
  b_size = hyp["b_size"]
  losses = [[] for i in range(len(classifiers))]

  filt_shared = [True if i in [0, 1, 2, 3] else False for i in y_train]

  shared_dataset = create_dataset(X_train[filt_shared], y_train[filt_shared])

  filt_oc1 = [True if i in [4] else False for i in y_train]
  filt_oc2 = [True if i in [5] else False for i in y_train]
  filt_oc3 = [True if i in [6] else False for i in y_train]

  oc1_dataset = create_dataset(X_train[filt_oc1], y_train[filt_oc1])
  oc2_dataset = create_dataset(X_train[filt_oc2], y_train[filt_oc2])
  oc3_dataset = create_dataset(X_train[filt_oc3], y_train[filt_oc3])

  datasets = [ torch.cat((shared_dataset, oc1_dataset[:numb_of_ood[i]], oc2_dataset[:numb_of_ood[i]], oc3_dataset[:numb_of_ood[i]]))
  for i in range(len(numb_of_ood))]

  dataloaders = [torch.utils.data.DataLoader(datasets[i], batch_size=b_size, shuffle=True, drop_last=True, worker_init_fn=0) for i in
  range(len(datasets))]
  data = torch.zeros((b_size, 2), dtype=torch.float).to(device)
  label = torch.zeros((b_size), dtype=int).to(device)

  pbar = tqdm([i for i in range(num_epoch)], total=num_epoch)

  for epoch in pbar:
    for i in range(len(classifiers)):
      for data_i in dataloaders[i]:
        data[:, :] = data_i[:, :2]
        label[:] = data_i[:, 2]

        classifiers[i].zero_grad()

        emb, out = classifiers[i](data)
        loss = loss_fn(out, label)
        losses[i].append(loss)

        loss.backward()

        optimizers[i].step()

    mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

    s = "%10s" % mem
    for i in range(len(classifiers)):
      s = s + "%10.4g" % losses[i][-1]
    pbar.set_description(s)

  test_dataset = torch.zeros((X_test.shape[0],3))
  test_dataset[:,:2] = torch.tensor(X_test,dtype=torch.float)
  test_dataset[:,2] = torch.tensor(y_test,dtype=int)

  confusion_matrixs = [test_function(classifiers[i],test_dataset,device) for i in range(len(classifiers))]

  accs = [acc_calc(confusion_matrixs[i]) for i in range(len(classifiers))]

  all_accs = [accs[i][0] for i in range(len(classifiers))]
  OoD_accs = [accs[i][1] for i in range(len(classifiers))]

  for i in range(len(numb_of_ood)):
    print("Accuracy of All for",numb_of_ood[i] ,":",'{0:.3g}'.format(all_accs[i]))
    print("Accuracy of OoD for",numb_of_ood[i] ,":",'{0:.3g}'.format(OoD_accs[i]))


  f = plt.figure(figsize=(10,10))

  plt.plot(numb_of_ood,all_accs,"-b",label = "All")
  plt.plot(numb_of_ood,OoD_accs,"--b",label = "OoD")
  plt.legend()
  plt.title("Accuracy-Number of OoD Images Curve")
  plt.xlabel("Number of Shared Images")
  plt.ylabel("Accuracy")
  plt.show()
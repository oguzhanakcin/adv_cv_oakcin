import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as ds
import os, random
import torchvision.datasets as datasets
import torch
import torch.optim as optim
import torchvision.transforms as trfm
import yaml
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn as nn
import cv2
from PIL import Image

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

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm1d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)

def load_ood_data(data_loc,name):
    train_ood_scores = np.load(data_loc +"/"+ name+"_train_ood_unc.npy")
    train_ind_scores = np.load(data_loc +"/"+ name+ "_train_ind_unc.npy")
    test_ood_scores = np.load(data_loc + "/" + name + "_test_ood_unc.npy")
    test_ind_scores = np.load(data_loc + "/" + name + "_test_ind_unc.npy")

    return train_ood_scores, train_ind_scores , test_ood_scores ,test_ind_scores

def load_sl_ood_data(data_loc,name,sim_type):

    train_ood_scores = np.load(data_loc +"/"+ name+"_"+sim_type+"_train_ood_unc.npy")
    train_ind_scores = np.load(data_loc +"/"+ name+ "_"+sim_type+"_train_ind_unc.npy")
    test_ood_scores = np.load(data_loc + "/" + name + "_"+sim_type+"_test_ood_unc.npy")
    test_ind_scores = np.load(data_loc + "/" + name + "_"+sim_type+"_test_ind_unc.npy")

    return train_ood_scores, train_ind_scores , test_ood_scores ,test_ind_scores

class MNISTDataset(Dataset):
    def __init__(self,X,y,transform=None):
        self.data = X.clone().detach().float().reshape((-1, 1, 28, 28))
        self.label = y.clone().detach().int()
        self.transform = transform

    def __len__(self):
        return self.label.size(0)

    def __getitem__(self,index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        y = self.label[index]

        return (x,y)

def create_labels_Wpred(data_loc,out_loc,train_ratio):
    f = []
    for (dirpath,dirnames,filenames) in os.walk(data_loc):
        f.extend(filenames)
    sunrise_imgs = [i  for i in f if i.startswith("sunrise")]
    cloudy_imgs = [i  for i in f if i.startswith("cloudy")]
    rain_imgs = [i for i in f if i.startswith("rain")]
    shine_imgs = [i for i in f if i.startswith("shine")]
    sunrise_train, sunrise_test = train_test_split(sunrise_imgs,random_state=1,train_size=train_ratio)
    cloudy_train, cloudy_test = train_test_split(cloudy_imgs, random_state=1, train_size=train_ratio)
    rain_train, rain_test = train_test_split(rain_imgs, random_state=1, train_size=train_ratio)
    shine_train, shine_test = train_test_split(shine_imgs, random_state=1, train_size=train_ratio)
    train_files = dict()
    for i in sunrise_train:
        train_files[i] = 0
    for i in cloudy_train:
        train_files[i] = 1
    for i in rain_train:
        train_files[i] = 2
    for i in shine_train:
        train_files[i] = 3
    test_files = dict()
    for i in sunrise_test:
        test_files[i] = 0
    for i in cloudy_test:
        test_files[i] = 1
    for i in rain_test:
        test_files[i] = 2
    for i in shine_test:
        test_files[i] = 3

    with open(out_loc+"/train.yaml","w") as f:
        yaml.dump(train_files,f)
    with open(out_loc+"/test.yaml","w") as f:
        yaml.dump(test_files,f)

class WheatherDataset(Dataset):
    def __init__(self,img_locs,dataset_loc,labels,transform=None):
        self.img_locs = img_locs
        self.labels = labels
        self.transform = transform
        self.root_dir = dataset_loc

    def __len__(self):
        return len(self.img_locs)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,self.img_locs[index])
        image = cv2.imread(img_name)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img).permute(2, 0, 1).float()
        label = self.labels[index]
        if self.transform:
            image = self.transform(img)
        return (image,label)

class RoadNetDataset(Dataset):
    def __init__(self,img_locs,dataset_loc,labels,transform=None):
        self.img_locs = img_locs
        self.labels = labels
        self.transform = transform
        self.root_dir = dataset_loc

    def __len__(self):
        return len(self.img_locs)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,self.img_locs[index])
        image = cv2.imread(img_name)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img).permute(2, 0, 1).float()
        label = self.labels[index]
        if label == "sunny":
            label = 0
        elif label == "overcast":
            label = 1
        elif label == "rainy":
            label = 2
        elif label == "snow":
            label = 3

        if self.transform:
            image = self.transform(img)
        return (image,label)

def create_RoadNet_dataset(img_locs,labels,dataset_locs):
    transform = trfm.Compose([trfm.Resize(256),
                              trfm.RandomCrop(224),
                              trfm.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    dataset = RoadNetDataset(img_locs,dataset_locs,labels,transform)
    return dataset

def create_Weather_dataset(img_locs,labels,dataset_locs):
    transform = trfm.Compose([trfm.Resize(256),
                              trfm.RandomCrop(224),
                              trfm.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    dataset = WheatherDataset(img_locs,dataset_locs,labels,transform)
    return dataset

def create_Weather_dataloader(dataset, b_size, device):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=b_size, shuffle=True, drop_last=True, worker_init_fn=0)
    data = torch.zeros((b_size,3, 224,224), dtype=torch.float).to(device)
    label = torch.zeros((b_size), dtype=int).to(device)
    return dataloader, data, label

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


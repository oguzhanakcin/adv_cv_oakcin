import torch
import torch.nn as nn
from nn_ood.data.rotated_mnist import RotatedMNIST
from nn_ood.posteriors import LocalEnsemble, Ensemble, SCOD, KFAC, Naive
from nn_ood.distributions import GaussianFixedDiagVar
import numpy as np
    
# WHERE TO SAVE THE MODEL
FILENAME = "model"

## HYPERPARAMS
N_MODELS = 5

LEARNING_RATE = 0.001
SGD_MOMENTUM = 0.9

LR_DROP_FACTOR = 0.5
EPOCHS_PER_DROP = 5

BATCH_SIZE = 16

N_EPOCHS = 50

## SET UP DATASETS
dataset_class = RotatedMNIST
test_dataset_args = ['val', 'ood', 'ood_angle']

## Dataset visualization
def plt_image(ax, inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = 0.1307
    std = 0.3081
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp[:,:,0], cmap='Greys')
    
def viz_dataset_sample(ax, dataset, idx=0, model=None, unc_model=None):
    input, target = dataset[idx]
    plt_image(ax, input)
    xlabel = 'Target: %0.2f' % target.item()
    if unc_model is not None:
        input = input.to(device)
        pred, unc = unc_model(input.unsqueeze(0))
        pred = pred[0].item()
        unc = unc.item()
        xlabel += '\nPred: %02f\nUnc: %0.3f' % (pred, unc)
    elif model is not None:
        input = input.to(device)
        pred = model(input.unsqueeze(0))[0]
        xlabel += '\nPred: %0.3f' % pred.item()
     
    ax.set_xlabel(xlabel)

def viz_datasets(idx=0, unc_model=None, model=None):
    num_plots = len(test_dataset_args)
    fig, axes = plt.subplots(1,num_plots, figsize=[5*num_plots, 5], dpi=100)
    for i, split in enumerate( test_dataset_args ):
        dataset = dataset_class(split)
        viz_dataset_sample(axes[i], dataset, idx=idx, unc_model=unc_model, model=model)

## USE CUDA IF POSSIBLE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## MODEL SET UP
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=5./3)
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)

def make_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(288, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    model.apply(weight_init)
    
    return model

def freeze_model(model, freeze_frac=True):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # make linear layers trainable
    for p in model.children[-2].parameters():
        p.requires_grad = True
    for p in model.children[-1].parameters():
        p.requires_grad = True
    
def unfreeze_model(model):
    # unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

dist_fam = GaussianFixedDiagVar().to(device)
opt_class = torch.optim.SGD
opt_kwargs = {
    'lr': LEARNING_RATE,
    'momentum': SGD_MOMENTUM
}
sched_class = torch.optim.lr_scheduler.StepLR
sched_kwargs = {
    'step_size': EPOCHS_PER_DROP,
    'gamma': LR_DROP_FACTOR
}    
    
prep_unc_models = {
    'scod_SRFT_s64_n10': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 64,
            'num_eigs': 10,
            'device':'gpu',
            'sketch_type': 'srft'
        },
    },
    'scod_SRFT_s154_n25': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 154,
            'num_eigs': 25,
            'device':'gpu',
            'sketch_type': 'srft'
        },
    },
    'scod_SRFT_s304_n50': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 304,
            'num_eigs': 50,
            'device':'gpu',
            'sketch_type': 'srft'
        },
    },
    'scod_SRFT_s604_n100': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 604,
            'num_eigs': 100,
            'device':'gpu',
            'sketch_type': 'srft'
        },
    },
}

keys_to_compare = []
colors = []

base_colors = [
    np.array([0.0, 1.0, 0.0, 0.0]),
    np.array([0.3, 0.7, 0.0, 0.0]),
    np.array([0.7, 0.3, 0.0, 0.0]),
    np.array([1.0, 0.0, 0.0, 0.0]),
]

test_unc_models = {}
for i,T in enumerate([64,154,304,604]):
    for r in [1,2,5,10,25,50,100,200,400]:
        num_eigs = int((T-4)/6)
        if r > 4*(T-4)/6:
            break
        exp_name = 'T=%d,r=%d' % (T,r)
        keys_to_compare.append(exp_name)
        exp = {
            'class': SCOD,
            'kwargs': {
                'num_samples': T,
                'num_eigs': num_eigs,
                'device':'gpu'
            },
            'load_name': 'scod_SRFT_s%d_n%d' % (T,num_eigs),
            'forward_kwargs': {
               'n_eigs': r
            }
        }
        
        color = base_colors[i]
        color[3] = 0.2 + 0.8*(r /( 4*num_eigs))
        color = color.tolist()
        
        colors.append(color)
        
        test_unc_models[exp_name] = exp

# OOD PERFORMANCE TESTS
splits_to_use = test_dataset_args
err_thresh = 1.

in_dist_splits = ['val']
out_dist_splits = ['ood','ood_angle']

# Results Visualization
from nn_ood.utils.viz import summarize_ood_results
import matplotlib.pyplot as plt

def plot_ablation(summarized_results):
    Ts = []
    rs = []
    aurocs = []
    auroc_confs = []
    for key,stats in summarized_results.items():
        tokenized = key.replace(',',' ').replace('=',' ').split(' ')
        T = int(tokenized[1])
        r = int(tokenized[3])
        Ts.append(T)
        rs.append(r)
        aurocs.append(stats['auroc'])
        auroc_confs.append(stats['auroc_conf'])

    Ts = np.array(Ts)
    rs = np.array(rs)
    aurocs = np.array(aurocs)
    auroc_confs = np.array(auroc_confs)
    
    plt.figure(figsize=[4,2.5],dpi=150)
    for i,T in enumerate(np.unique(Ts)):
        valid_idx = Ts == T
        plt.axvline((T-4)/6, color='C'+str(i), alpha=0.8, linestyle=':')
        plt.errorbar(rs[valid_idx], aurocs[valid_idx], 
                     yerr=auroc_confs[valid_idx], label='T=%d'%T,
                    linestyle='-', marker='', capsize=2)

    plt.legend()
    plt.xscale('log')
    plt.xlabel(r'Rank of approximation $k$')
    plt.ylabel('AUROC')
    
plots_to_generate = {
    'auroc_vs_rank.pdf': {
        'summary_fn': summarize_ood_results,
        'summary_fn_args': [
            in_dist_splits,
            out_dist_splits
        ],
        'summary_fn_kwargs': {
            'keys_to_compare': keys_to_compare,
        },
        'plot_fn': plot_ablation,
        'plot_fn_args': [],
        'plot_fn_kwargs': {},
    },
}
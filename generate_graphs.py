import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_data(data_loc,st):
    n_imgs = np.load(data_loc + "/" + st + "_nimg.npy")
    ood_accs = np.load(data_loc + "/" + st + "_ood_acc.npy")
    all_accs = np.load(data_loc + "/" + st + "_all_acc.npy")
    return n_imgs,ood_accs,all_accs


def generate_graph(opt):

    n_imgs = dict()
    ood_accs = dict()
    all_accs = dict()
    if opt.plot_random :
        n_imgs["random"] , ood_accs["random"] , all_accs["random"] = load_data(opt.data_loc,"random")
        
    if opt.plot_oracle :
        n_imgs["oracle"] , ood_accs["oracle"] , all_accs["oracle"] = load_data(opt.data_loc, "oracle")

    if opt.plot_soft :
        n_imgs["softmax"] , ood_accs["softmax"] , all_accs["softmax"] = load_data(opt.data_loc, "softmax")

    if opt.plot_entr :
        n_imgs["entropy"] , ood_accs["entropy"] , all_accs["entropy"] = load_data(opt.data_loc, "entropy")

    if opt.plot_scod:
        n_imgs["scod"] , ood_accs["scod"] , all_accs["scod"] = load_data(opt.data_loc, "scod")

    if opt.plot_gu:
        n_imgs["gu"] , ood_accs["gu"] , all_accs["gu"] = load_data(opt.data_loc, "gu")

    f = plt.figure(figsize=(10,10))

    for i in list(n_imgs.keys()):
        n_rounds = [i for i in range(len(n_imgs[i]))]
        plt.semilogy(n_rounds,n_imgs[i],label=i)

    plt.legend()
    plt.xlabel("Number of Rounds")
    plt.ylabel("Total Number of OoD Images")
    plt.savefig("shared_util.png")

    f = plt.figure(figsize=(10,10))

    for i in list(n_imgs.keys()):
        n_rounds = [i for i in range(len(n_imgs[i]))]
        plt.plot(n_rounds,all_accs[i],label=i)

    plt.legend()
    plt.xlabel("Number of Rounds")
    plt.ylabel("All Accuracy")
    plt.savefig("all_acc.png") 
   


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--plot-random", action="store_false")
    parser.add_argument("--plot-oracle", action="store_false")
    parser.add_argument("--plot-soft", action="store_false")
    parser.add_argument("--plot-entr", action="store_false")
    parser.add_argument("--plot-scod", action="store_false")
    parser.add_argument("--plot-gu", action="store_false")
    parser.add_argument("--gen-sim-graph", action= "store_false")
    parser.add_argument("--data-loc",type=str,default="./experiments/synthetic/sim_data")

    opt = parser.parse_args()
    if opt.gen_sim_graph:
        generate_graph(opt)


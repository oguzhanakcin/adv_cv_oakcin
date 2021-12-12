import matplotlib.pyplot as plt
import numpy as np
import argparse

# Function to load the graph data
def load_data(data_loc,st):
    n_imgs = np.load(data_loc + "/" + st + "_nimg.npy")
    ood_accs = np.load(data_loc + "/" + st + "_ood_acc.npy")
    all_accs = np.load(data_loc + "/" + st + "_all_acc.npy")
    return n_imgs,ood_accs,all_accs

# Function to plot graphs
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

    if opt.plot_gu:
        n_imgs["gu"] , ood_accs["gu"] , all_accs["gu"] = load_data(opt.data_loc, "gu")

    f = plt.figure(figsize=(10,10))

    for i in list(n_imgs.keys()):
        n_rounds = [i for i in range(len(n_imgs[i]))]
        res = [i for i in range(len(n_imgs[i]))]
        res[0] = n_imgs[i][0]
        for j in range(1,len(n_imgs[i])):
            res[j] = n_imgs[i][j] - n_imgs[i][j-1] 
        plt.plot(n_rounds,res,label=i)

    plt.legend()
    plt.xlabel("Number of Rounds")
    plt.ylabel("Total Number of OoD Images")
    plt.savefig("shared_util.png")

    f = plt.figure(figsize=(10,10), dpi=1200)
    plt.rcParams["font.size"]=18
    plt.rcParams["axes.linewidth"]=2
    

    for i in list(n_imgs.keys()):
        n_rounds = [i for i in range(len(n_imgs[i]))]
        plt.plot(n_rounds,all_accs[i],label=i,linewidth=4)

    plt.legend(["Random","Oracle","Softmax","Entropy","GU Model"])

    plt.xlabel("Training Episodes",fontweight="bold" ,fontsize=24)
    plt.ylabel("Accuracy",fontweight="bold" ,fontsize=24)
    plt.yticks([i/20 for i in range(5,16)])
    plt.grid(linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.savefig("all_acc.png") 

    f = plt.figure(figsize=(10,10))

    for i in list(n_imgs.keys()):
        n_rounds = [i for i in range(len(n_imgs[i]))]
        plt.plot(n_rounds,ood_accs[i],label=i)

    plt.legend()
    plt.xlabel("Number of Rounds")
    plt.ylabel("OoD Accuracy")
    plt.savefig("ood_acc.png") 

if __name__ == "__main__":
    # Plotting parameters 
    parser = argparse.ArgumentParser()

    parser.add_argument("--plot-random", action="store_false")
    parser.add_argument("--plot-oracle", action="store_false")
    parser.add_argument("--plot-soft", action="store_false")
    parser.add_argument("--plot-entr", action="store_false")
    parser.add_argument("--plot-gu", action="store_false")
    parser.add_argument("--gen-sim-graph", action= "store_false")
    parser.add_argument("--data-loc",type=str,default="./experiments/MNIST/sim_data")

    opt = parser.parse_args()
    if opt.gen_sim_graph:
        generate_graph(opt)

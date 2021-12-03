import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_data(data_loc,st):
    n_imgs = np.load(data_loc + "/" + st + "_nimg.npy")
    ood_accs = np.load(data_loc + "/" + st + "_ood_acc.npy")
    all_accs = np.load(data_loc + "/" + st + "_all_acc.npy")
    return n_imgs,ood_accs,all_accs

def load_ood_data(data_loc):
    ood_scores = np.load(data_loc + "/scod_ood_unc.npy")
    ind_scores = np.load(data_loc + "/scod_ind_unc.npy")
    return ood_scores,ind_scores

def generate_graph(opt):

    if opt.plot_random :
        random_n_imgs , random_ood_accs , random_all_accs = load_data(opt.data_loc,"random")

    if opt.plot_oracle :
        oracle_n_imgs, oracle_ood_accs, oracle_all_accs = load_data(opt.data_loc, "oracle")

    if opt.plot_soft :
        soft_n_imgs, soft_ood_accs, soft_all_accs = load_data(opt.data_loc, "softmax")

    if opt.plot_entr :
        entr_n_imgs, entr_ood_accs, entr_all_accs = load_data(opt.data_loc, "entropy")

    if opt.plot_scod:
        scod_n_imgs, scod_ood_accs, scod_all_accs = load_data(opt.data_loc, "scod")

    n_rounds = [i for i in range(len(random_n_imgs))]
    f = plt.figure(figsize=(10,10))

    if opt.plot_random:
        plt.semilogy(n_rounds,random_n_imgs,"-b",label="Random Sampling")

    if opt.plot_oracle:
        plt.semilogy(n_rounds,oracle_n_imgs, "-r", label="Oracle Sampling")

    if opt.plot_soft:
        plt.semilogy(n_rounds,soft_n_imgs,"-g",label="Softmax Sampling")

    if opt.plot_entr:
        plt.semilogy(n_rounds,entr_n_imgs,"-c",label="Entropy Sampling")

    if opt.plot_scod:
        plt.semilogy(n_rounds,scod_n_imgs,"-m",label="SCOD Sampling")

    plt.legend()
    plt.xlabel("Number of Rounds")
    plt.ylabel("Total Number of OoD Images")
    plt.savefig("shared_util.png")

def generate_ood_graph(opt):

    f = plt.figure(figsize=(10, 10))

    ood_scores, ind_scores = load_ood_data(opt.data_loc)

    data = [ind_scores,ood_scores]

    box = plt.boxplot(data, patch_artist=True, )
    plt.xticks([1, 2], ["In Distribution", "Out of Distribution"])
    plt.ylabel("Uncertainity Score")
    plt.xticks(rotation=45)
    colors = ["#0000FF", "#FF0000"]

    for i in range(len(box["boxes"])):
        box["boxes"][i].set_facecolor(colors[i])

    plt.savefig("ood_scores.png")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--plot-random", action="store_false")
    parser.add_argument("--plot-oracle", action="store_false")
    parser.add_argument("--plot-soft", action="store_false")
    parser.add_argument("--plot-entr", action="store_false")
    parser.add_argument("--plot-scod", action="store_false")
    parser.add_argument("--gen-sim-graph", action= "store_false")
    parser.add_argument("--gen-ood-score", action= "store_false")
    parser.add_argument("--data-loc",type=str,default="./experiments/synthetic/sim_data")

    opt = parser.parse_args()
    opt.plot_soft = False
    opt.plot_entr = False
    if opt.gen_sim_graph:
        generate_graph(opt)

    if opt.gen_ood_score:
        generate_ood_graph(opt)
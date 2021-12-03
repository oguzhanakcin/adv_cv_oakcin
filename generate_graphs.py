import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils import *

def normalize_uncs(data):
    s1,s2,s3,s4 = data[0],data[1],data[2],data[3]
    min_s = min(min(s1),min(s2),min(s3),min(s4))
    max_s = max(max(s1),max(s2),max(s3),max(s4))
    s1 = [(i - min_s) / (max_s - min_s) for i in s1]
    s2 = [(i - min_s) / (max_s - min_s) for i in s2]
    s3 = [(i - min_s) / (max_s - min_s) for i in s3]
    s4 = [(i - min_s) / (max_s - min_s) for i in s4]
    return s1,s2,s3,s4

def generate_ood_graph(opt):

    f = plt.figure(figsize=(10, 10))
    if opt.model_size:
        train_ood_scores, train_ind_scores, test_ood_scores, test_ind_scores = load_sl_ood_data(opt.data_loc, opt.ood_type,opt.model_size)
    else:
        train_ood_scores, train_ind_scores , test_ood_scores ,test_ind_scores = load_ood_data(opt.data_loc,opt.ood_type)

    data = [train_ind_scores ,test_ind_scores, train_ood_scores,  test_ood_scores ]

    box = plt.boxplot(data, patch_artist=True, )
    plt.xticks([1, 2,3,4], ["Train In Distribution", "Test In Distribution", "Train Out of Distribution","Test Out of Distribution"])
    plt.ylabel("Uncertainity Score")
    plt.xticks(rotation=45)
    colors = ["#0000FF","#0000FF", "#FF0000", "#FF0000"]

    for i in range(len(box["boxes"])):
        box["boxes"][i].set_facecolor(colors[i])
    if opt.model_size:
        plt.savefig(opt.data_loc+"/"+opt.ood_type+opt.model_size+"_ood_scores.png")
    else:
        plt.savefig(opt.data_loc + "/" + opt.ood_type + "_ood_scores.png")

def generate_multiple_graph(opt):
    f = plt.figure(figsize=(10, 10))
    if opt.multiple_small_large:
        s_train_ood, s_train_ind, s_test_ood, s_test_ind = load_sl_ood_data(opt.data_loc, opt.ood_type,"small")
        m_train_ood, m_train_ind, m_test_ood, m_test_ind = load_sl_ood_data(opt.data_loc, opt.ood_type, "medium")
        l_train_ood, l_train_ind, l_test_ood, l_test_ind = load_sl_ood_data(opt.data_loc,opt.ood_type,"large")

        s_train_ood, s_train_ind, s_test_ood, s_test_ind = normalize_uncs([s_train_ood, s_train_ind, s_test_ood, s_test_ind])
        m_train_ood, m_train_ind, m_test_ood, m_test_ind = normalize_uncs(
            [m_train_ood, m_train_ind, m_test_ood, m_test_ind])

        l_train_ood, l_train_ind, l_test_ood, l_test_ind = normalize_uncs(
            [l_train_ood, l_train_ind, l_test_ood, l_test_ind])


        data = [s_train_ind,m_train_ind,l_train_ind,s_test_ind,m_test_ind,l_test_ind,
                s_train_ood,m_train_ood,l_train_ood,s_test_ood,m_test_ood,l_test_ood]
    else:
        if opt.model_size:
            soft_train_ood, soft_train_ind, soft_test_ood, soft_test_ind = load_sl_ood_data(opt.data_loc,
                                                                                                    "Softmax",
                                                                                                    opt.model_size)
            entr_train_ood, entr_train_ind, entr_test_ood, entr_test_ind = load_sl_ood_data(opt.data_loc,
                                                                                                    "Entropy",
                                                                                                    opt.model_size)
            scod_train_ood, scod_train_ind, scod_test_ood, scod_test_ind = load_sl_ood_data(opt.data_loc,
                                                                                                    "Scod",
                                                                                                    opt.model_size)
        else:
            soft_train_ood, soft_train_ind, soft_test_ood, soft_test_ind = load_ood_data(opt.data_loc,
                                                                                            "Softmax")
            entr_train_ood, entr_train_ind, entr_test_ood, entr_test_ind = load_ood_data(opt.data_loc,
                                                                                            "Entropy")
            scod_train_ood, scod_train_ind, scod_test_ood, scod_test_ind = load_ood_data(opt.data_loc,
                                                                                            "Scod")

        scod_train_ood, scod_train_ind, scod_test_ood, scod_test_ind = normalize_uncs(
            [scod_train_ood, scod_train_ind, scod_test_ood, scod_test_ind])
        entr_train_ood, entr_train_ind, entr_test_ood, entr_test_ind = normalize_uncs(
            [entr_train_ood, entr_train_ind, entr_test_ood, entr_test_ind])
        soft_train_ood, soft_train_ind, soft_test_ood, soft_test_ind = normalize_uncs(
            [soft_train_ood, soft_train_ind, soft_test_ood, soft_test_ind])

        data = [soft_train_ind,entr_train_ind,scod_train_ind,soft_test_ind,entr_test_ind,scod_test_ind,soft_train_ood,entr_train_ood,scod_train_ood,
                soft_test_ood, entr_test_ood, scod_test_ood]

    if opt.multiple_small_large:
        box = plt.boxplot(data,widths = (0.2),
                      positions=[1,1.35,1.7, 2.5,2.85,3.2,4,4.35,4.7,5.5,5.85,6.2],
                      labels=['Small Train Ind','Medium Train Ind', 'Large Train Ind', 'Small Test Ind',
                              'Medium Test Ind','Large Test Ind', 'Small Train OoD','Medium Train Ood',
                              'Large Train OoD', 'Small Test OoD','Medium Test OoD', 'Large Test OoD'],
                          patch_artist=True)
        plt.ylabel("Uncertainity Score")
        plt.xticks(rotation=45)
        colors = ["#0000FF", "#FF0000", "#00FF00", "#0000FF", "#FF0000", "#00FF00", "#0000FF", "#FF0000", "#00FF00",
                  "#0000FF", "#FF0000", "#00FF00"]

        for i in range(len(box["boxes"])):
            box["boxes"][i].set_facecolor(colors[i])
        plt.savefig(opt.data_loc + "/" + opt.ood_type + "_s_m_l_comp.png")
    else:
        box = plt.boxplot(data,widths = (0.2),
                      positions=[1,1.35,1.7, 2.5,2.85,3.2,4,4.35,4.7,5.5,5.85,6.2],
                      labels=["Softmax Train Ind","Entropy Train Ind","SCOD Train Ind","Softmax Test Ind","Entropy Test Ind","SCOD Test Ind",
                              "Softmax Train OoD","Entropy Train OoD","SCOD Train OoD","Softmax Test OoD","Entropy Test OoD","SCOD Test OoD"],
                          patch_artist=True)
        plt.ylabel("Uncertainity Score")
        plt.xticks(rotation=45)
        colors = ["#0000FF", "#FF0000","#00FF00","#0000FF", "#FF0000","#00FF00","#0000FF", "#FF0000","#00FF00","#0000FF", "#FF0000","#00FF00"]
        for i in range(len(box["boxes"])):
            box["boxes"][i].set_facecolor(colors[i])
        if opt.model_size:
            plt.savefig(opt.data_loc + "/" + opt.model_size + "unc_comp.png")
        else:
            plt.savefig(opt.data_loc + "/" + "unc_comp.png")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ood-type",type=str,default="Softmax")
    parser.add_argument("--generate_multiple", action= "store_true")
    parser.add_argument("--multiple_small_large", action="store_true")
    parser.add_argument("--data-loc",type=str,default="./experiments/RoadNet/sim_data")
    parser.add_argument("--model-size", type=str, default=None)

    opt = parser.parse_args()
    opt.generate_multiple = True
    opt.multiple_small_large = True
    opt.ood_type = "Scod"

    if opt.generate_multiple:
        generate_multiple_graph(opt)
    else:
        generate_ood_graph(opt)
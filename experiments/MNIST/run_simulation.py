import argparse
import yaml
import os, sys
import torch
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from experiments.MNIST import *

def run_sim(hyp,opt,device):

    X_train,y_train, X_test,y_test = load_MNIST_dataset(opt.dataset_loc,opt.num_samples)

    data = [X_train,y_train, X_test,y_test]

    if opt.train_base:
        print("Training base model")
        train_base(data,device,hyp["base_model"],hyp["general"])

    if opt.sim_random:
        print("Simulating random model")
        simulate(data,device,hyp["random_model"],hyp["general"],opt.sim_result_loc,"random")

    if opt.sim_oracle:
        print("Simulating oracle model")
        simulate(data,device,hyp["oracle_model"],hyp["general"],opt.sim_result_loc,"oracle")

    if opt.sim_soft:
        print("Simulating softmax model")
        simulate(data,device,hyp["soft_model"],hyp["general"],opt.sim_result_loc,"softmax")

    if opt.sim_entropy:
        print("Simulating Entropy model")
        simulate(data, device, hyp["entropy_model"],hyp["general"], opt.sim_result_loc, "entropy")


    if opt.sim_gu:
        print("Simulating Gu model")
        simulate(data, device, hyp["gu_model"],hyp["general"], opt.sim_result_loc, "gu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-loc", type=str,default="./data")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--hyp-loc",type=str,default="hyp.yaml")
    parser.add_argument("--train-base", action="store_true")
    parser.add_argument("--sim-random",action="store_true")
    parser.add_argument("--sim-oracle",action="store_true")
    parser.add_argument("--sim-soft",action="store_true")
    parser.add_argument("--sim-entropy",action="store_true")
    parser.add_argument("--sim-gu", action="store_true")
    parser.add_argument("--sim-result-loc", type=str, default="sim_data")

    opt = parser.parse_args()

    with open(opt.hyp_loc) as f:
        hyp = yaml.load(f,Loader=yaml.SafeLoader)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print('Using torch %s %s' % (
    torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

    run_sim(hyp,opt,device)
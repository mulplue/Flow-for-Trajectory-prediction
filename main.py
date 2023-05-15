import argparse
import pickle
import warnings
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm, trange

from utils.utils import setup_seed
from model.flow import FlowNet
from utils.dataloader import SceneDataset, scene_collate




## Argparser
parser = argparse.ArgumentParser(description="AF")
# data config
parser.add_argument("--dataset", type=str, default="eth")
parser.add_argument("--scene", type=str, default="zara2")
# training config
parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("-b", "--batch_size", type=int, default=8)
# device config
parser.add_argument("-d", "--device", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--des", type=str, default="")
args = parser.parse_args()


## Setup
# cuda
torch.cuda.set_device(args.device)
setup_seed(args.seed)
# wandb
project_name = "AF-Zara2" 
exp_name = args.des
wandb.init(project=project_name, name=exp_name, config=args)
# config
config_file_path = "config/flow_try1.yaml"
with open(config_file_path) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# train func
def train(model, optimizer, params, epochs, batch_size):
    model = model.cuda()
    obs_len = params["obs_len"]
    pred_len = params["pred_len"]
    
    for e in trange(epochs):
        # train
        model.train()
        loss = 0
        log_prob_loss = 0
        logabsdet_loss = 0
        for batch, (trajectory, meta, scene) in enumerate(train_loader):    # scene
            for i in range(0, len(trajectory), batch_size):     # trajectory
                observed = trajectory[i : i + batch_size, :obs_len, :].cuda()      # (batch_size, obs_len, 2)
                gt_future = trajectory[i : i + batch_size, obs_len:].cuda()
                log_prob, logabsdet = model(observed, gt_future)
                # Compute NLL Loss
                nll_loss = - torch.mean(log_prob - logabsdet, dim=0)
                nll_loss.backward()
                # Gradient Step
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                
                loss += nll_loss.item()
                log_prob_loss += log_prob.mean().item()
                logabsdet_loss += logabsdet.mean().item()
                
        # evaluate
        model.eval()
        val_ADE = []
        val_FDE = []
        with torch.no_grad():
            for batch, (trajectory, meta, scene) in enumerate(val_loader):
                for i in range(0, len(trajectory), batch_size):
                    observed = trajectory[i : i + batch_size, :obs_len, :].cuda()       # (batch_size, obs_len, 2)
                    gt_future = trajectory[i : i + batch_size, obs_len:].cuda()
                    gt_goal = gt_future[:, -1:]
                    
                    sampled_trajs = model.predict_n(observed, params["diverse"])
            #         val_FDE.append(
            #             (
            #                 (((gt_goal - sampled_trajs[:, -1:]) / params["resize"]) ** 2).sum(
            #                     dim=3
            #                 )
            #                 ** 0.5
            #             ).min(dim=0)[0]
            #         )
            #         val_ADE.append(
            #             ((((gt_future - sampled_trajs) / params["resize"]) ** 2).sum(dim=3) ** 0.5)
            #             .mean(dim=2)
            #             .min(dim=0)[0]
            #         )
            # val_ADE = torch.cat(val_ADE).mean()
            # val_FDE = torch.cat(val_FDE).mean()
            # print(f"Epoch {e}: \nVal ADE: {val_ADE} \nVal FDE: {val_FDE}")
                    


if __name__ == "__main__":
    ## Load data
    # data path
    train_data_path = "data/eth_ucy/pickles/ynet_processed/zara2_train.pkl"
    val_data_path = "data/eth_ucy/pickles/ynet_processed/zara2_test.pkl"
    homo_mat_path = "data/eth_ucy/homo_mat/"

    # data loader
    df_train = pd.read_pickle(train_data_path)
    df_val = pd.read_pickle(val_data_path)

    train_dataset = SceneDataset(
        df_train, resize=params['resize'], total_len=params['total_len'],
    )
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=scene_collate,)
    val_dataset = SceneDataset(
        df_val, resize=params["resize"], total_len = params['total_len']
    )
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=scene_collate)


    ## Load model
    model_args = params["AF"]
    model = FlowNet(**model_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])


    ## Train model
    train(model, optimizer, params, args.epochs, args.batch_size)
















# input()
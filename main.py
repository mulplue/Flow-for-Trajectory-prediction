import argparse
import pickle
import warnings
import math
import cv2
import numpy as np
import pandas as pd
import torch
import random
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm, trange

from utils.utils import setup_seed, image2world, get_homoMat
from utils.visualization import plot_traj_static, create_images_dict
from model.flow import FlowNet
from utils.dataloader import SceneDataset, scene_collate


# train func
def train(
    train_loader, val_loader,
    model, optimizer,
    params,
    epochs, batch_size,
    dataset_name="eth",
    homo_mat_path="data/eth_ucy/homo_mat/",
    semantic_map_path="data/eth_ucy/semantic_map",
    exp_name="test",
):
    model = model.cuda()
    wandb.watch(model)
    
    obs_len = params["obs_len"]
    pred_len = params["pred_len"]
    homo_mat = get_homoMat(homo_mat_path)
    semantic_map, rgb_map = create_images_dict(semantic_map_path)        # a dict, (scene name, 2 class semantic map)

    best_val_ADE = 999
    best_val_FDE = 999
    for e in trange(epochs):
        # train
        model.train()
        loss = 0
        log_prob_loss = 0
        logabsdet_loss = 0
        for batch, (trajectory, meta, scene) in enumerate(train_loader):  # scene
            for i in range(0, len(trajectory), batch_size):  # trajectory
                observed = trajectory[
                    i : i + batch_size, :obs_len, :
                ].cuda()  # (B, obs_len, 2)
                gt_future = trajectory[i : i + batch_size, obs_len:].cuda()
                log_prob, logabsdet = model(observed, gt_future)
                # Compute NLL Loss
                nll_loss = -torch.mean(log_prob - logabsdet, dim=0)
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
            visual_scene = 0
            for batch, (trajectory, meta, scene) in enumerate(val_loader):
                traj_list = np.arange(0, len(trajectory), batch_size)
                visual_traj = random.choice(traj_list)
                visual_batch = 0
                for i in range(0, len(trajectory), batch_size):
                    observed = trajectory[
                        i : i + batch_size, :obs_len, :
                    ].cuda()  # (B, obs_len, 2)
                    gt_future = trajectory[
                        i : i + batch_size, obs_len:
                    ].cuda()  # (B*n, pred_len, 2)

                    pred_trajs = model.predict_n(
                        observed, params["diverse"]
                    )  # (B*n, pred_len, 2)
                    pred_trajs = pred_trajs.reshape(params["diverse"], -1, pred_len, 2)     # (n, B, pred_len, 2)
                    
                    # visualize in pixel coordinate
                    if batch == visual_scene and i == visual_traj:
                        fig = plot_traj_static(
                            observed[visual_batch],
                            future=gt_future[visual_batch],
                            prediction=pred_trajs[:, visual_batch],
                            num=5,
                            map=rgb_map[scene],
                        )
                        # plt.savefig("test.png")

                    # compute ADE & FDE in world coordinate
                    if dataset_name == "eth":
                        observed = image2world(observed, scene, homo_mat, 1)
                        pred_trajs = image2world(pred_trajs, scene, homo_mat, 1)
                        gt_future = image2world(gt_future, scene, homo_mat, 1)

                    val_ADE.append(
                        (
                            (((gt_future - pred_trajs) / params["resize"]) ** 2).sum(
                                dim=3
                            )
                            ** 0.5
                        )
                        .mean(dim=2)
                        .min(dim=0)[0]
                    )       # item: (B, )
                    val_FDE.append(
                        (
                            (
                                (
                                    (gt_future[:, -1:] - pred_trajs[:, :, -1:])
                                    / params["resize"]
                                )
                                ** 2
                            ).sum(dim=3)
                            ** 0.5
                        ).min(dim=0)[0]
                    )       # item: (B, 1)


            val_ADE = torch.cat(val_ADE).mean()
            val_FDE = torch.cat(val_FDE).mean()
            print(f"Epoch {e}: \nVal ADE: {val_ADE} \nVal FDE: {val_FDE}")

        # log
        if val_ADE < best_val_ADE:
            print(f"Best Epoch {e}: \nVal ADE: {val_ADE} \nVal FDE: {val_FDE}")
            torch.save(model.state_dict(), "checkpoint/" + exp_name + ".pt")
            best_val_ADE = val_ADE
        if val_FDE < best_val_FDE:
            best_val_FDE = val_FDE

        wandb.log(
            {
                "epoch": e,
                "nll_loss": loss,
                "logprob_loss": log_prob_loss,
                "logabsdet_loss": logabsdet_loss,
                "val_ADE": val_ADE,
                "val_FDE": val_FDE,
                "best_val_ADE": best_val_ADE,
                "best_val_FDE": best_val_FDE,
                "visual": wandb.Image(fig),
            }
        )


if __name__ == "__main__":
    ## Argparser
    parser = argparse.ArgumentParser(description="AF")
    # data config
    parser.add_argument("--dataset", type=str, default="eth")
    parser.add_argument("--scene", type=str, default="zara2")
    # training config
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    # device config
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--des", type=str, default="")
    args = parser.parse_args()

    ## Setup
    # cuda
    torch.cuda.set_device(args.device)
    setup_seed(args.seed)
    # config
    config_file_path = "config/flow_try1.yaml"
    with open(config_file_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    # wandb
    project_name = "AF-Zara2"
    exp_name = args.des
    wandb.init(
        project=project_name,
        name=exp_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "description": args.des,
        },
    )
    

    ## Load data
    # data path
    train_data_path = "data/eth_ucy/pickles/ynet_processed/zara2_train.pkl"
    val_data_path = "data/eth_ucy/pickles/ynet_processed/zara2_test.pkl"
    homo_mat_path = "data/eth_ucy/homo_mat/"
    # img_file_path = ""

    # data loader
    df_train = pd.read_pickle(train_data_path)
    df_val = pd.read_pickle(val_data_path)

    train_dataset = SceneDataset(
        df_train,
        resize=params["resize"],
        total_len=params["total_len"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=scene_collate,
    )
    val_dataset = SceneDataset(
        df_val, resize=params["resize"], total_len=params["total_len"]
    )
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=scene_collate)

    ## Load model
    model_args = params["AF"]
    model = FlowNet(**model_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ## Train model
    train(
        train_loader, val_loader,
        model, optimizer,
        params,
        args.epochs, args.batch_size,
        exp_name = args.des,
    )
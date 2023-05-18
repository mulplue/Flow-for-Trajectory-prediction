import torch
import random
import numpy as np
import torch.nn.functional as F


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def get_homoMat(homo_mat_path):
    homo_mat = {}
    for scene in [
        "eth",
        "hotel",
        "students001",
        "students003",
        "uni_examples",
        "zara1",
        "zara2",
        "zara3",
    ]:
        homo_mat[scene] = torch.Tensor(
            np.loadtxt(homo_mat_path + f"{scene}_H.txt")
        ).cuda()
    return homo_mat
    
    
def image2world(image_coords, scene, homo_mat, resize):
    """
    Transform trajectories of one scene from image_coordinates to world_coordinates
    :param image_coords: torch.Tensor, shape=[num_person, (optional: num_samples), timesteps, xy]
    :param scene: string indicating current scene, options=['eth', 'hotel', 'student01', 'student03', 'zara1', 'zara2']
    :param homo_mat: dict, key is scene, value is torch.Tensor containing homography matrix (data/eth_ucy/scene_name.H)
    :param resize: float, resize factor
    :return: trajectories in world_coordinates
    """
    traj_image2world = image_coords.clone()
    if traj_image2world.dim() == 4:
        traj_image2world = traj_image2world.reshape(-1, image_coords.shape[2], 2)
    if scene in ["eth", "hotel"]:
        # eth and hotel have different coordinate system than ucy data
        traj_image2world[:, :, [0, 1]] = traj_image2world[:, :, [1, 0]]
    traj_image2world = traj_image2world / resize
    traj_image2world = F.pad(
        input=traj_image2world, pad=(0, 1, 0, 0), mode="constant", value=1
    )
    traj_image2world = traj_image2world.reshape(-1, 3)
    traj_image2world = torch.matmul(homo_mat[scene], traj_image2world.T).T
    traj_image2world = traj_image2world / traj_image2world[:, 2:]
    traj_image2world = traj_image2world[:, :2]
    traj_image2world = traj_image2world.view_as(image_coords)
    return traj_image2world
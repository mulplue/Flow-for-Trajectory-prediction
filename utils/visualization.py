import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

color = ['b', 'c', 'g', 'm', 'y']

## some utils
def tensor2array(tensor):
    if torch.is_tensor(tensor):
        try:
            tensor = tensor.detach().cpu().numpy()
        except:
            tensor = tensor.cpu().numpy()
    return tensor


def plot_traj_utils(ax, traj, c1, c2):
    traj = tensor2array(traj)
    ax.scatter(traj[0, 0], traj[0, 1], marker=".", s=20, color=c1)
    ax.scatter(traj[1:, 0], traj[1:, 1], marker=".", s=10, color=c2)
    return ax


## plot trajectory
def set_fig(title=None):
    fig = plt.figure()
    plt.axis("off")
    if title is not None:
        plt.title(title)
    else:
        plt.title("")
    return fig


def plot_traj_static(observed, future=None, prediction=None, num = None, map=None, title=None):
    observed = tensor2array(observed)
    fig = set_fig(title)
    ax = fig.add_subplot(111)
    if map is not None:
        ax.imshow(map)

    plot_traj_utils(ax, observed, c1="red", c2="black")
    if future is not None:
        future = tensor2array(future)
        plot_traj_utils(ax, future, c1="orange", c2="orange")
    if prediction is not None:
        prediction = tensor2array(prediction)
        if num is None:
            num = len(prediction)
        for pi in range(num):
            plot_traj_utils(ax, prediction[pi], c1=color[pi], c2=color[pi])
    return fig


## map
def create_images_dict(image_path, image_file='oracle.png'):
    semantic = {}
    rgb = {}
    if image_file == 'oracle.png':
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
            im_semantic = cv2.imread(os.path.join(image_path, scene, image_file), 0)
            im_rgb = semantic2rgb(im_semantic)
            semantic[scene] = im_semantic
            rgb[scene] = im_rgb
    return semantic, rgb


def semantic2rgb(semantic_image):
    semantic_image = tensor2array(semantic_image)
    semantic_image = np.expand_dims(semantic_image, -1).repeat(3, axis=-1)
    
    semantic_image[semantic_image == 0] = 0
    semantic_image[semantic_image == 1] = 255
    return semantic_image
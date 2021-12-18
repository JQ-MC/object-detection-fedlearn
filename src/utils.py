import os
import glob
import torch
from torch import nn
from torch.utils.data import DataLoader
import shutil

import yaml
from pathlib import Path

import random
import logging
from distutils.dir_util import copy_tree

from utils.downloads import attempt_download
from utils.torch_utils import (
    intersect_dicts,
    torch_distributed_zero_first,
)
from utils.general import increment_path

import val

import pandas as pd
import numpy as np
import uuid

from typing import List

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
LOGGER = logging.getLogger(__name__)
from models.yolo import Model


def get_size_after_cnn(h: int, k: int, s: int, p: int):
    "calculates the size of the images after 2 convolutions"

    size = -(-(h - k + s + p)) / s
    size = -(-size / 2)
    size = -(-(size - k + s + p)) / s
    size = -(-size / 2)

    return size


def eval_net(opt):
    """
    Function that implements the evaluation of the model according to Yolo but adapting
    it to Flower structure

    Args:
    -------------------
    opt: runner_server.py options

    Return
    -------------------
    results: results of evaluation
    """

    save_dir = increment_path(Path("runs/eval/") / opt.name)
    os.mkdir(save_dir)

    results, _, _ = val.run(
        data=opt.data,
        model=creation_of_the_model(opt),
        device=opt.device,
        save_dir=save_dir,
        save_model=True,
    )

    return results


def creation_of_the_model(opt):
    """the model is created. All config variables are extracted from a config file
    (originally). Now they are written ans instanciated here"""

    # weights = find_weights_for_model()
    # found all in input parameters (opt)
    resume = opt.resume
    weights = str(opt.weights)
    device = opt.device
    cfg = opt.cfg
    hyp = opt.hyp
    with open(hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)
    with open(opt.data, errors="ignore") as f:
        nc = yaml.safe_load(f)
        nc = int(nc["nc"])

    # Model
    # check_suffix(weights, ".pt")  # check weights
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(
            cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")
        ).to(
            device
        )  # create
        exclude = (
            ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []
        )  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(
            f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}"
        )  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create

    return model


def folder_data_splitter_iid(
    folder_origin: str, n_splits, agent, img_extension="png", ext="_iid", seed=13
) -> None:
    """
    Creates an iid partition of a dataset that follows Yolov5 indications to make a federated
    learning

    Given a folder path, it recreates the same one with the suffix `ext`, selects the
    partition that corresponds to the agent and deletes the other images

    Args:
    folder_origin: folder to clone and perform tasks
    n_splits: number of agents implicated. If there are two partitions, n_splits=2
    agent: int. Must be between [0, n_splits]. Number of one of the partitions done in n_splits
    img_extension: images ectension
    ext: suffix to append to folder_origin and create a new one
    seed: seed.
    """

    random.seed(seed)

    # Creation of the dest file
    new_folder = folder_origin + ext

    # In case the folder exists, delete and re-create
    try:
        os.mkdir(new_folder)
    except:
        shutil.rmtree(new_folder)
        os.mkdir(new_folder)

    # copy dependecies from folder_name to new_file
    copy_tree(folder_origin, new_folder)

    # train folder
    train_new_folder = new_folder + "/train"

    # List all images in the folder
    img_path_list = glob.glob(f"{train_new_folder}/*." + img_extension)

    # Generate uniform distribution of the images
    num_imgs = len(img_path_list)
    randomlist = [random.randint(1, n_splits) for x in range(num_imgs)]

    print(randomlist.count(agent))

    # Assign each image a group and keep those of a different group
    indices = [i for i, x in enumerate(randomlist) if x != agent]

    # Delete the images and annotation that do not correspond with agent
    for index in indices:
        os.remove(img_path_list[index])
        os.remove(img_path_list[index][:-3] + "txt")


def folder_data_splitter_non_iid(
    folder_origin, n_splits, agent, img_extension="png", ext="_non_iid", seed=13
) -> None:
    """
    Creates a non iid partition of a dataset that follows Yolov5 indications to make a federated
    learning

    Given a folder path, it recreates the same one with the suffix `ext`, selects the
    partition that corresponds to the agent and deletes the other images

    Args:
    folder_origin: folder to clone and perform tasks
    n_splits: number of agents implicated. If there are two partitions, n_splits=2
    agent: int. Must be between [0, (n_splits-1)]. Number of one of the partitions to keep in n_splits
    img_extension: images ectension
    ext: suffix to append to folder_origin and create a new one
    seed: seed.
    """

    # check if agent is < 3

    random.seed(seed)
    np.random.seed(seed)

    # Creation of the dest file
    new_folder = folder_origin + ext

    # In case the folder exists, delete and re-create
    try:
        os.mkdir(new_folder)
    except:
        shutil.rmtree(new_folder)
        os.mkdir(new_folder)

    # copy dependecies from folder_name to new_file
    copy_tree(folder_origin, new_folder)

    # train folder
    train_new_folder = new_folder + "/train"

    # List all images in the folder
    img_path_list = glob.glob(f"{train_new_folder}/*." + img_extension)

    # ------------  Create non iid distribution

    # Collect all groups in different lists. the groups have their name differenciated
    groups = create_groups(img_path_list)  # for each object

    # Partition interval created
    cut = np.linspace(-2, 2, num=(n_splits + 1))[:-1]
    bins = [-np.inf]
    for x in cut[1:]:
        bins.append(x)
    bins.append(np.inf)

    names = [i for i in range(n_splits)]
    for group in groups:  # per a cada grup

        random.shuffle(names)
        print(names)

        # randomly generate a set of numbers following a normal distribution to
        # define the partition groups
        res = np.random.normal(loc=0.0, scale=1.0, size=len(group))
        res2 = pd.cut(res, bins, labels=names)

        # Delete the images and annotation that do not correspond with agent
        for index in range(len(res2)):
            if res2[index] != agent:

                os.remove(group[index])
                os.remove(group[index][:-3] + "txt")

    img_path_list = glob.glob(f"{train_new_folder}/*." + img_extension)
    print(len(img_path_list))


def create_groups(img_path_list: List) -> List:
    """
    splits into groups a list of images path. Images must be named according to their group

    Args:
    ----------------
    img_path_list: list of images paths

    Return:
    ----------------
    result_list: list of lists containing the splitted groups
    """

    img_path_list.sort()

    images_name = [img.split("/")[-1] for img in img_path_list]

    route = img_path_list[0].split("/")[:-1]
    route = "/".join(route)

    img_check = images_name[0][0:4]
    group_list = []
    result_list = []
    for img in images_name:

        if img[0:4] != img_check:
            result_list.append(group_list)
            group_list = [route + "/" + img]
            img_check = img[0:4]
        else:
            group_list.append(route + "/" + img)

    result_list.append(group_list)

    return result_list


def split_dataset_val_train(
    path: str, img_ext: str = "png", train_split: float = 0.75
) -> None:
    """
    Splits a dataset of images into train and val

    Args
    --------------
    path: path where the folder is
    img_ext: extension of the images
    train_split: percentage of the train set

    """

    # get all images
    img_path_list = glob.glob(f"{path}/*." + img_ext)

    # Shuffle
    random.shuffle(img_path_list)

    # Train Set
    for img_path in img_path_list[0 : int(len(img_path_list) * train_split)]:

        img_path_png = img_path
        img_path_txt = img_path[0 : len(img_path) - 3] + "txt"

        dst_path = os.path.dirname(path) + "/train/"
        shutil.move(img_path_png, dst_path)
        shutil.move(img_path_txt, dst_path)

    # Test set
    img_path_list = glob.glob(f"{path}/*." + img_ext)
    for img_path in img_path_list:

        img_path_png = img_path
        img_path_txt = img_path[0 : len(img_path) - 3] + "txt"

        dst_path = os.path.dirname(path) + "/val/"
        shutil.move(img_path_png, dst_path)
        shutil.move(img_path_txt, dst_path)


def rename_images(path: str, dest_file: str, name: str, img_ext: str = "png") -> None:
    """
    Renames images of a folder in this way:
    `name` + (12 random caracters) + `img_ext`

    Args:
    -------------------
    path: object path where the images are
    dest_path: path of the folder where the images will be moved
    name: name to add to the images
    img_ext: extension of the images
    """

    # Listing images of path
    img_path_list = glob.glob(f"{path}/*." + img_ext)

    # Moving and rename
    for img_path in img_path_list:
        dst_path = dest_file + "/" + name + str(uuid.uuid4())[-12:] + "." + img_ext
        shutil.move(img_path, dst_path)


if __name__ == "__main__":

    folder_origin = (
        "/Users/joaquimmorera/Desktop/fedlearning_yolo/data/datasets/camacuc2"
    )
    folder_data_splitter_non_iid(folder_origin=folder_origin, n_splits=3, agent=1)

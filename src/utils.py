import os
import torch
from torch import nn
from torch.utils.data import DataLoader

import yaml
from pathlib import Path

import os
import logging


from utils.downloads import attempt_download
from utils.torch_utils import (
    intersect_dicts,
    torch_distributed_zero_first,
)
from utils.general import increment_path

import val

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


def train_net(model, dataloader: DataLoader, epochs: int) -> None:
    "training of the models"
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    model.train()

    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(model.device), y.to(model.device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        print()
        print(f"Epoch: {epoch+1} / {epochs}")


def test_net(opt):

    save_dir = increment_path(Path("runs/eval/") / opt.name)
    os.mkdir(save_dir)

    results, _, _ = val.run(
        data=opt.data,
        model=creation_of_the_model(opt),
        device=opt.device,
        save_dir=save_dir,
    )

    return results


def load_model(model_str):
    "returns selected model"

    if model_str == "BasicCNN":
        return BasicCNN(in_chn=1, n_out=10)
    elif model_str == "FullyConnected":
        return FullyConnected(n_inp=1, n_out=10)


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

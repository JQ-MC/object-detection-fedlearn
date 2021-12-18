from collections import OrderedDict
import torchvision.transforms as transforms
import flwr as fl
import torch

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights
from flwr.common.logger import log
from logging import INFO
from src.utils import test_net, creation_of_the_model
import timeit
import numpy as np
import os
import logging

from train import train_handler

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
LOGGER = logging.getLogger(__name__)


def get_weights(model: torch.nn.Module) -> fl.common.Weights:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: torch.nn.Module, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=False)


class YOLO_Client(fl.client.Client):
    def __init__(self, cid, opt) -> None:
        super().__init__()
        self.model = creation_of_the_model(opt)
        self.cid = cid
        self.opt = opt

    def get_parameters(self) -> ParametersRes:
        log(INFO, f"Client {self.cid}: get_parameters")

        weights: Weights = get_weights(self.model)
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        log(INFO, f"Client {self.cid}: fit")

        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        batch_size = int(config["batch_size"])
        pin_memory = bool(config["pin_memory"])
        num_workers = int(config["num_workers"])

        # Set model parameters
        log(INFO, f"Client {self.cid}: set_weights")
        set_weights(self.model, weights)

        if torch.cuda.is_available():
            kwargs = {
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "drop_last": True,
            }
        else:
            kwargs = {"drop_last": True}

        # Train model
        log(INFO, f"Client {self.cid}: train_net")
        train_handler(model=self.model, opt=self.opt)
        log(INFO, f"Client {self.cid}: finish train_net")
        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = get_weights(self.model)
        params_prime = fl.common.weights_to_parameters(weights_prime)
        # num_examples_train = len(trainloader.dataset)
        num_examples_train = 1
        metrics = {"duration": timeit.default_timer() - fit_begin}
        return FitRes(
            parameters=params_prime, num_examples=num_examples_train, metrics=metrics
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        log(INFO, f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        set_weights(self.model, weights)

        # Evaluate the updated model on the local dataset
        testloader = FashionMNIST_test(4, transformation=transforms.ToTensor())
        loss, accuracy = test_net(self.model, testloader)

        # Return the number of evaluation examples and the evaluation result (loss)
        metrics = {"accuracy": float(accuracy)}
        return EvaluateRes(
            num_examples=len(testloader.dataset), loss=float(loss), metrics=metrics
        )

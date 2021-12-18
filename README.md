# Federated Learning: Object detection with YOLOv5

Repository of the conceptual work around Federated Learning to train the object detection model YOLOv5. 

Here you can find the developed code, tests, concept experiments and demos.

The current repository implements the logic regarding the tasks that the server and clients do.

## Clone repository

```
git clone -b develop https://user:password@gitlab.com/cecotec/ia/miscelaneous/fl-object-detection fedlearn
cd fedlearn
pip3 install -r requirements.txt
```


### venv

In order to activate `venv` environment, execute the following command from repository root:
```
source venv/bin/activate
```

## Repository folder structure:
- `.vscode`: vscode configuration.
- `data`: datasets and configuration files about the datasets.
- `models`: implemented models and configuration files about the models.
- `src`: source code.
    - `client`: implements the logic of the client in FL
    - `server`: implements the logic of the server in FL
    - `strategy`: implements the strategies that guide the server and the parameters aggregation
    - `utils`: custom helper functions
- `data_splitter.py`: Splits a folder of images that follow YOLOv5 guidelines into partitions to try a federated approach to train a model
- `yolo_train_solo.py`: trains yolo in a centralized way
- `runner_client.py`: sets up a client to train the net
- `runner_server.py`: sets up the server to handle the clients
- `utils`: yolo helper functions.
- `detect.py`: used to try the model in the webcam or images
- `train.py`: implements the train pipeline of the model. Adapted to be used through the clients
- `val.py`: implements the validation of the model. Adapted to be used through the server.


## SetUp
Set up a Federated Learning process!
#### Server

In order to start a Federated Learning training, it is necessary to start the server:

`python3 runner_server.py --server_address [::]:8080 --rounds 2 --loads_params`

```
- server_address: ip of the server
- rounds: number of federated rounds
- loads_params: controls if the server provides the initial parameters or it gets them from a random client (`--no_load_params`)
```
**Default useful parameters:**
```
- weights: initial weights path. `default=ROOT / "yolov5s.pt"`
- cfg: path to the specifications about the structure of the model to be trained. default="models/yolov5s.yaml"
- hyp: path to the specifications about the hyperparameters of the model. default=ROOT / "data/camacuc2.yaml"
- data: path to the specifications about the data to be trained. default=ROOT / "data/hyps/hyp.scratch.yaml"
```

**Control the number of clients that participate in a Federated training**
```
- min_num_clients: Minimum number of available clients required for sampling (default: 2)
- min_sample_size: Minimum number of clients used for fit/evaluate (default: 2)
- sample_fraction: Fraction of available clients used for fit/evaluate (default: 1.0
```

#### Client

Clients must be created after the server. There must be at least `--min_num_clients` clients, a parameter from the server, to start the training. Defaults to 2.

`python3 runner_client.py --server_address [::]:8080 --cid 1 --epochs 100`

```
- server_address: ip of the server. if executed in the same machine keep default. If not, check ip address of the server: XXX.XX.X.XX:8080
- cid: client id
- epochs: number of epochs per round
```
**Default useful parameters:**
```
- weights: initial weights path. `default=ROOT / "yolov5s.pt"`
- cfg: path to the specifications about the structure of the model to be trained. default="models/yolov5s.yaml"
- hyp: path to the specifications about the hyperparameters of the model. default=ROOT / "data/camacuc2.yaml"
- data: path to the specifications about the data to be trained. default=ROOT / "data/hyps/hyp.scratch.yaml"
```

**Memory Handling:**

In order to reduce the amount of memory in RAM and specially for embedded devices, lowering these parameters may help

```
- img: train, val image size (pixels)
- batch-size: number of images to pass through each batch of an epoch
``` 

## Federated Object Detection on the Edge

This repository is able to run on edge devices such as Raspberry Pi 4 or Pi 3B.

In order to flash the raspberries follow these tutorials:
* Install 64 bit OS (Tested Bullseye): https://qengineering.eu/install-raspberry-64-os.html
* Install Pytorch and Pytorchvision wheels: https://qengineering.eu/install-pytorch-on-raspberry-pi-4.html

Then clone the repository and install requirements with pip3.

It is very important to handle RAM memory. 

## Run YOLOv5 centralized

You can run YOLOv5 as the original repository:

```
$ python yolo_train_solo.py --data data/camacuc2.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                                            yolov5m                                40
                                                                                                   24
                                                                                                   16
```

### Future work

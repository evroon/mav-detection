# optical-flow-mav-detection

![preview](https://ci.tno.nl/gitlab/erik.vroon-tno/optical-flow-mav-detection/-/raw/master/media/preview.png)

This project is intended to run on Ubuntu 20.04, with Python 3.8.

## Install
Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Set the settings for AirSim:

```bash
cp etc/settings.json ~/Documents/AirSim/settings.json
```

The following environment variables need to be set in `.bashrc` with their correct paths:
```bash
export DATASET_DIR="~/tno/datasets/drone-tracking-datasets/dataset2/cam1"
export FLOWNET2="~/flownet2-pytorch"
export FLOWNET2_CHECKPOINTS_DIR="~/tno/datasets/flownet2-checkpoints"
export MIDGARD_PATH="~/tno/datasets/MIDGARD"
export SIMDATA_PATH="~/tno/datasets/sim-data"
export YOLOv4_PATH="~/tno/yolov4"
export UNDISTORT_PATH="~/tno/datasets/undistortFunctions/launch_docker.sh"
```

## Usage
To see all possible command arguments:

```bash
python3 src/main.py --help
```

To check the typing of the Python code:

```bash
mypy
```

## YOLOv4

![test](https://github.com/evroon/yolov4/workflows/docker-build/badge.svg)

## RAFT

![test](https://github.com/evroon/RAFT/workflows/docker-build/badge.svg)

## FlowNet2

![test](https://github.com/evroon/flownet2-pytorch/workflows/docker-build/badge.svg)

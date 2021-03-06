# Motion-based MAV detection
[![docker-build](https://github.com/evroon/mav-detection/actions/workflows/main.yml/badge.svg)](https://github.com/evroon/mav-detection/actions/workflows/main.yml)

![preview](media/preview.png)

This project performs motion-based object detection of MAVs using optical flow and the Focus of Expansion. It also includes Python scripts to generate datasets from AirSim. It is intended to run on Ubuntu 20.04, with Python 3.8.

## Install
Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Set the settings for AirSim:

```bash
cp etc/settings.json ~/Documents/AirSim/settings.json
```

You will need the `FlowNet2` and `semantic-segmentation` forks listed under [dependencies](#dependencies-for-object-detection).
The following environment variables need to be set in `.bashrc` with their correct paths:
```bash
export FLOWNET2="~/neural-nets/flownet2-pytorch"
export FLOWNET2_CHECKPOINTS_DIR="~/neural-nets/flownet2-checkpoints"
export HRNET_PATH="~/neural-nets/semantic-segmentation"
```

Optionally, you can set the following environment variables depending on which dataset you want to use:
```bash
export SIMDATA_PATH="~/datasets/sim-data"
export MIDGARD_PATH="~/datasets/midgard"
export EXPERIMENT_PATH="~/datasets/experiment"
```

## Usage
To see all possible command arguments:

```bash
python3 src/main.py --help
```

Which outputs:
```
usage: main.py [-h] [--dataset DATASET] [--sequence SEQUENCE] [--mode MODE] [--algorithm ALGORITHM] [--debug] [--prepare-dataset] [--validate] [--headless] [--run-all] [--data-to-yolo] [--undistort]

Detects MAVs in the dataset using optical flow.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset to process
  --sequence SEQUENCE   sequence to process
  --mode MODE           mode to use, see RunConfig.Mode
  --algorithm ALGORITHM
                        detection algorithm to use, see Detection.Algorithm
  --debug               whether to debug or not
  --prepare-dataset     prepares the YOLOv4 training dataset
  --validate            validate the detection results
  --headless            do not use UIs
  --run-all             run all configurations
  --data-to-yolo        convert annotations to the YOLO format
  --undistort           undistort original
```


To check the typing of the Python code run:

```bash
mypy
```

## Dependencies for object detection
### [flownet2-pytorch](https://github.com/evroon/flownet2-pytorch)
[![docker-build](https://github.com/evroon/flownet2-pytorch/actions/workflows/main.yml/badge.svg)](https://github.com/evroon/flownet2-pytorch/actions/workflows/main.yml)

### [semantic-segmentation](https://github.com/evroon/semantic-segmentation)
[![docker-build](https://github.com/evroon/semantic-segmentation/actions/workflows/main.yml/badge.svg)](https://github.com/evroon/semantic-segmentation/actions/workflows/main.yml)

## To create datasets with AirSim
### [AirSim](https://github.com/evroon/AirSim)
[![Ubuntu Build](https://github.com/evroon/AirSim/actions/workflows/test_ubuntu.yml/badge.svg)](https://github.com/evroon/AirSim/actions/workflows/test_ubuntu.yml)

## Other repositories created for this project
### [YOLOv4](https://github.com/evroon/yolov4)
[![docker-build](https://github.com/evroon/yolov4/actions/workflows/main.yml/badge.svg)](https://github.com/evroon/yolov4/actions/workflows/main.yml)

### [raft](https://github.com/evroon/RAFT)
[![docker-build](https://github.com/evroon/RAFT/actions/workflows/main.yml/badge.svg)](https://github.com/evroon/RAFT/actions/workflows/main.yml)

### [undistort-functions](https://github.com/evroon/undistort-functions)
[![test](https://github.com/evroon/undistort-functions/actions/workflows/main.yml/badge.svg)](https://github.com/evroon/undistort-functions/actions/workflows/main.yml)

### [mask-rcnn](https://github.com/evroon/mask-rcnn)
[![test](https://github.com/evroon/mask-rcnn/actions/workflows/main.yml/badge.svg)](https://github.com/evroon/mask-rcnn/actions/workflows/main.yml)

### [docker-orb-slam2-build](https://github.com/evroon/docker-orb-slam2-build)

### [pytorch-liteflownet](https://github.com/evroon/pytorch-liteflownet)

### [MaskFlownet](https://github.com/evroon/MaskFlownet)

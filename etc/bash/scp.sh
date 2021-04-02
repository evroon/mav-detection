#!/bin/bash
FRAME=127

scp -r erik@192.168.178.235:/home/erik/tno/datasets/MIDGARD/indoor-modern/sports-hall .

scp -r /mnt/c/Users/Daniel/Documents/uni/thesis/optical-flow-mav-detection/data/* erik@192.168.178.235:~/tno/datasets/sim-data

scp -r /mnt/d/UnrealProjects/LandscapeMountains/Builds/LinuxNoEditor/ erik@192.168.178.235:~/tno/UE4/LandscapeMountains

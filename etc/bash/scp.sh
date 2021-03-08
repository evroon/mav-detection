#!/bin/bash
FRAME=127

scp -r erik@192.168.178.235:/home/erik/tno/datasets/MIDGARD/indoor-modern/sports-hall .

sequence="test/test"
scp -r /mnt/c/Users/Daniel/Documents/uni/thesis/images erik@192.168.178.235:~/tno/datasets/sim-data/$sequence
scp -r /mnt/c/Users/Daniel/Documents/uni/thesis/segmentations erik@192.168.178.235:~/tno/datasets/sim-data/$sequence
scp -r /mnt/c/Users/Daniel/Documents/uni/thesis/states erik@192.168.178.235:~/tno/datasets/sim-data/$sequence

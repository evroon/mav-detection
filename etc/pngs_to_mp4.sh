#!/bin/bash

ffmpeg -r 30 -i image_%05d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p output.mp4 -y
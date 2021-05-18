#!/bin/bash

ffmpeg -i recording.mp4 -filter:v "crop=2048:1350:0:186" recording-cropped.mp4

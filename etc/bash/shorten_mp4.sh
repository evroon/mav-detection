#!/bin/bash

ffmpeg -ss 00:04:54.0 -i recording.mp4 -c copy -t 00:00:15.0 recording-sample.mp4

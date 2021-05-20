#!/bin/bash

# Save only every fourth frame.
ffmpeg -i recording.mp4 -vf select='not(mod(n\,4)), setpts=0.25*PTS' -an recording-low-fps.mp4

#!/bin/bash
FRAME=127

for x in $(ls *.mp4); do ffmpeg -i $x -vf select='between(n\,$FRAME\,$FRAME)' -vsync 0 frames/${x}_%d.png; done
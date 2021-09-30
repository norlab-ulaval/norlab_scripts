#! /bin/bash

if [ $# -ne 3 ]
then
	printf "This script creates a video from a serie of images.\n"
	printf "Invalid number of arguments. Arguments needed:\n 1: Path to the folder containing the frames;\n 2: Video name (you need to add .mp4 at the end);\n 3: Frame rate.\n"
	exit
fi

folder_path=$1
video_name=$2
frame_rate=$3

ffmpeg -framerate $frame_rate -pattern_type glob -i "$folder_path/*.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 $video_name
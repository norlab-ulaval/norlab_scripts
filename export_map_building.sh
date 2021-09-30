#! /bin/bash

if [ $# -ne 4 ]
then
  echo "Usage: Records the map as it gets built. The bag file needs to be played with the --pause argument and the mapper needs to be running."
  echo "Incorrect number of arguments. Argument 1 is the total length of the bag. Argument 2 is the wanted video length. Argument 3 is the wanted frame rate. Argument 4 is the output folder."
  exit 1
fi

bag_length=$1
video_length=$2
frame_rate=$3
output_folder=$4

(( nb_frames = video_length * frame_rate ))
(( delay_between_saves = bag_length / nb_frames ))

pause_service_name=`rosservice list | grep "pause_playback"`

for i in $(seq 1 $nb_frames)
  do
    rosservice call $pause_service_name "data: false"
    sleep $delay_between_saves
    rosservice call $pause_service_name "data: true"
    rosservice call /save_map "map_file_name:
    data: '$output_folder/map_$i.vtk'"
    rosservice call /save_trajectory "trajectory_file_name:
    data: '$output_folder/trajectory_$i.vtk'"
  done

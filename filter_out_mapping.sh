#! /bin/bash

if [ $# -ne 2 ]; then
  echo "Invalid number of arguments. Argument 1 is the input bag file. Argument 2 is the output bag file."
  exit 1
fi

rosbag filter $1 $2 "topic != '/map' and topic != '/icp_odom' and (topic != '/tf' or (m.transforms[0].header.frame_id != 'map' and m.transforms[0].header.frame_id != '/map') or (m.transforms[0].child_frame_id != 'odom' and m.transforms[0].child_frame_id != '/odom'))"


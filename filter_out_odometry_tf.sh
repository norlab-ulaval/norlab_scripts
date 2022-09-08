#! /bin/bash

if [ $# -ne 2 ]; then
  echo "Invalid number of arguments. Argument 1 is the input bag file. Argument 2 is the output bag file."
  exit 1
fi

rosbag filter $1 $2 "topic != '/tf' or (m.transforms[0].header.frame_id != 'odom' and m.transforms[0].header.frame_id != '/odom') or (m.transforms[0].child_frame_id != 'base_link' and m.transforms[0].child_frame_id != '/base_link' and m.transforms[0].child_frame_id != 'body' and m.transforms[0].child_frame_id != '/body' and m.transforms[0].child_frame_id != '/base_link_stabilized')"


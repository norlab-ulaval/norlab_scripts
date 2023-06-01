#! /usr/bin/python3

import sys
import yaml


if len(sys.argv) != 2:
    print("Incorrect number of arguments! Argument 1 is the path to the metadata file of the bag to fix.")
    exit(1)

with open(sys.argv[1], "r") as bag_metadata_file:
    bag_metadata = yaml.safe_load(bag_metadata_file)
for topic in bag_metadata["rosbag2_bagfile_information"]["topics_with_message_count"]:
    msg_type = topic["topic_metadata"]["type"]
    msg_type_name_start_position = msg_type.rfind("/") + 1
    if msg_type[msg_type_name_start_position].islower():
        topic["topic_metadata"]["type"] = msg_type[:msg_type_name_start_position] + msg_type[msg_type_name_start_position].upper() + msg_type[msg_type_name_start_position+1:]
with open(sys.argv[1], "w") as bag_metadata_file:
    yaml.dump(bag_metadata, bag_metadata_file)


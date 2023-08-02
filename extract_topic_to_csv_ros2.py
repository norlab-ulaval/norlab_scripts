#! /usr/bin/python3

import sys
import os
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import generate_msgdef
from tqdm import tqdm

def get_field_names(parent_field_type, parent_field_name=None):
    if '/' in parent_field_type:
        if '/msg/' not in parent_field_type:
            parent_field_type = parent_field_type[0:parent_field_type.rfind('/')] + "/msg" + parent_field_type[parent_field_type.rfind('/'):]
        field_names = []
        fields = generate_msgdef(parent_field_type)[0].split("\n")[:-1]
        for field in fields:
            if field == "================================================================================":
                break
            field_type = field.split(' ')[0]
            if field_type == "time": # this happens when using bags converted from ROS 1
                field_type = "builtin_interfaces/Time"
            field_name = field.split(' ')[1]
            if parent_field_name is not None:
                field_name = parent_field_name + "." + field_name
            field_names.extend(get_field_names(field_type, field_name))
        return field_names
    else:
        return [parent_field_name]

def export_topic_cmd(argv):
    bag_file_name, csv_file_name, topic = argv

    if not os.path.isdir(bag_file_name):
        print('Error: Cannot locate input bag file [%s]' % bag_file_name, file=sys.stderr)
        sys.exit(2)

    with Reader(bag_file_name) as bag_file, open(csv_file_name, "w+") as csv_file:
        field_names = []
        csv_header_written = False
        progress_bar = None
        progress_bar_initial_count = 0
        for conn, timestamp, data in bag_file.messages():
            if conn.topic == topic:
                try:
                    msg = deserialize_cdr(data, conn.msgtype)
                    if not csv_header_written:
                        csv_header = ""
                        for field_name in get_field_names(conn.msgtype):
                            try:
                                field_value = eval("msg." + field_name)
                                field_names.append(field_name)
                                if csv_header != "":
                                    csv_header += ","
                                csv_header += field_name
                            except AttributeError:
                                pass
                                print("Warning: Cannot retrieve field " + field_name + " of desired topic.")
                                continue
                        csv_file.write(csv_header + "\n")
                        csv_header_written = True
                        progress_bar = tqdm(total=bag_file.message_count, initial=progress_bar_initial_count)

                    csv_line = ""
                    for field_name in field_names:
                        if csv_line != "":
                            csv_line += ","
                        csv_line += str(eval("msg." + field_name))
                    csv_file.write(csv_line.replace("\n", "") + "\n")
                except:
                    print("Error: Unable to deserialize messages from desired topic.")
                    sys.exit(3)
            if progress_bar is not None:
                progress_bar.update()
            else:
                progress_bar_initial_count += 1

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Error: Incorrect number of arguments.\n\tArgument 1 is the input bag to extract a topic from.\n\tArgument 2 is the output csv file name.\n\tArgument 3 is the topic name to extract.")
        sys.exit(1)

    export_topic_cmd(sys.argv[1:])


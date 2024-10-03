#! /usr/bin/python3.10

import sys
from typing import TYPE_CHECKING, cast
from rosbags.interfaces import ConnectionExtRosbag2
from rosbags.rosbag2 import Reader, Writer
from rosbags.serde import deserialize_cdr, serialize_cdr


def filter(src, dst) -> None:
    with Reader(src) as reader, Writer(dst) as writer:
        conn_map = {}
        for conn in reader.connections:
            if conn.topic == "/mapping/map" or conn.topic == "/mapping/icp_odom":
                continue
            ext = cast(ConnectionExtRosbag2, conn.ext)
            try:
                conn_map[conn.id] = writer.add_connection(
                    conn.topic,
                    conn.msgtype,
                    serialization_format=ext.serialization_format,
                    offered_qos_profiles=ext.offered_qos_profiles,
                )
            except:
                print("The message ", conn.topic, " will not be written to the new bag, the definition is unknown.")
                continue

        for conn, timestamp, data in reader.messages():
            if conn.topic == "/mapping/map" or conn.topic == "/mapping/icp_odom":
                continue
            if conn.topic == "/tf":
                msg = deserialize_cdr(data, conn.msgtype)
                if (
                    msg.transforms[0].header.frame_id == "map"
                    and msg.transforms[0].child_frame_id == "odom"
                ):
                    continue
            try:
                writer.write(conn_map[conn.id], timestamp, data)
            except:
                continue


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Invalid number of arguments. Argument 1 is the input bag. Argument 2 is the output bag."
        )
        exit(1)
    else:
        filter(sys.argv[1], sys.argv[2])


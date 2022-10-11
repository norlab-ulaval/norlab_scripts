#! /usr/bin/python3

import sys
import os
from rosbags.rosbag2 import Reader, Writer
from rosbags.serde import deserialize_cdr
from typing import cast
from rosbags.interfaces import ConnectionExtRosbag2

def filter_cmd(argv):
    def expr_eval(expr):
        return eval("lambda topic, m, t: %s" % expr)

    inbag_filename, outbag_filename, expr = argv

    if not os.path.isdir(inbag_filename):
        print('Cannot locate input bag file [%s]' % inbag_filename, file=sys.stderr)
        sys.exit(2)

    if os.path.realpath(inbag_filename) == os.path.realpath(outbag_filename):
        print('Cannot use same file as input and output [%s]' % inbag_filename, file=sys.stderr)
        sys.exit(3)

    filter_fn = expr_eval(expr)

    with Reader(inbag_filename) as inbag, Writer(outbag_filename) as outbag:
        conn_map = {}
        for conn, timestamp, data in inbag.messages():
            try:
                msg = deserialize_cdr(data, conn.msgtype)
                if filter_fn(conn.topic, msg, timestamp):
                    if conn.id not in conn_map:
                        ext = cast(ConnectionExtRosbag2, conn.ext)
                        conn_map[conn.id] = outbag.add_connection(
                            conn.topic,
                            conn.msgtype,
                            ext.serialization_format,
                            ext.offered_qos_profiles,
                        )
                    outbag.write(conn_map[conn.id], timestamp, data)
            except KeyError:
                print("Warning: Cannot find definition of message type %s. Discarding message..." % conn.msgtype)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Incorrect number of arguments.\n\tArgument 1 is the input bag to filter.\n\tArgument 2 is the output bag.\n\tArgument 3 is the filtering Python expression. The variable 'topic' is the topic of the message, 'm' is the message and 't' is the time of the message.")
        exit(1)

    filter_cmd(sys.argv[1:])


#! /bin/bash

if [ "$#" -ne 3 ]; then
	echo "Incorrect number of arguments. Argument 1 is the bag file containing the topic to extract. Argument 2 is the topic to extract. Argument 3 is the output csv file name."
	exit 1
fi

rostopic echo -b "$1" -p "$2" > "$3"

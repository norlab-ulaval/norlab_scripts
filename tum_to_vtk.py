import argparse
import os

import numpy as np
import pandas as pd
from pypointmatcher import pointmatcher as pm, pointmatchersupport as pms
from os.path import basename


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main_file(filepath: str):
    if not filepath.endswith(".csv"):
        return
    filepath_out = filepath.replace("csv", "vtk")
    filepath_out = filepath_out.replace("trajectories", "vtk")
    try:
        df = pd.read_csv(filepath, sep=" ", names=["timestamp", "x", "y", "z", "q_x", "q_y", "q_z", "q_w"])
        print(f"{bcolors.OKGREEN}Converting {basename(filepath)} to {basename(filepath_out)}{bcolors.ENDC}")

        DP = pm.PointMatcher.DataPoints
        ptcloud = DP()
        ptcloud.features = np.vstack((df["x"], df["y"], df["z"], np.ones(len(df["x"]))))
        ptcloud.save(filepath_out)
        print(f"{bcolors.OKGREEN}...Finished {filepath_out}{bcolors.ENDC}")
    except FileNotFoundError as e:
        print(f"{bcolors.WARNING}Can't find {filepath}: {e}{bcolors.ENDC}")


def main_path(path: str):
    files = os.listdir(path)
    for file in sorted(files):
        filepath = path + file
        main_file(filepath)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path",
                        type=str,
                        help="Folder with trajectory files in the tum format:"
                             "timestamp, x, y, z, q_x, q_y, q_z, q_w columns.")
    parser.add_argument("-f", "--file",
                        type=str,
                        help="Trajectory file in the tum format:"
                             "timestamp, x, y, z, q_x, q_y, q_z, q_w columns.")
    return parser


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    assert args.path is not None or args.file is not None
    if args.path is not None:
        main_path(args.path)
    if args.file is not None:
        main_file(args.file)

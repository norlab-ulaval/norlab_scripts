import argparse
import pandas as pd
from scipy.spatial.transform import Rotation as R


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


def main(input_filepath, output_filepath):
    print(f"{bcolors.OKGREEN}Processing {input_filepath}...{bcolors.ENDC}")
    input_data = pd.read_csv(input_filepath)
    timestamps = input_data["ros_time"]*1e-9
    quaternion = R.from_euler("xyz", input_data[["roll", "pitch", "yaw"]].values, degrees=False).as_quat()
    df = pd.DataFrame({"timestamp": timestamps,
                       "x": input_data["x"],
                       "y": input_data["y"],
                       "z": input_data["z"],
                       "q_x": quaternion[:, 0],
                       "q_y": quaternion[:, 1],
                       "q_z": quaternion[:, 2],
                       "q_w": quaternion[:, 3]})
    df.to_csv(output_filepath, index=False, sep=" ", header=None)
    print(f"{bcolors.OKGREEN}...Finished, saved to {output_filepath}{bcolors.ENDC}")


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        type=str, required=True,
                        help=".csv file containing dumped ROS2 pose/odom messages.")
    parser.add_argument("-o", "--output",
                        type=str, required=True,
                        help=".csv file in tum format"
                             "(Contains: timestamp, x, y, z, q_x, q_y, q_z, q_w in its columns).")
    return parser


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    main(args.input, args.output)

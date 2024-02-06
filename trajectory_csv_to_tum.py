import argparse
import pandas as pd


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
    timestamps = input_data["header.stamp.sec"] + input_data["header.stamp.nanosec"] / 1e9
    df = pd.DataFrame({"timestamp": timestamps,
                       "x": input_data["pose.pose.position.x"],
                       "y": input_data["pose.pose.position.y"],
                       "z": input_data["pose.pose.position.z"],
                       "q_x": input_data["pose.pose.orientation.x"],
                       "q_y": input_data["pose.pose.orientation.y"],
                       "q_z": input_data["pose.pose.orientation.z"],
                       "q_w": input_data["pose.pose.orientation.w"]})
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

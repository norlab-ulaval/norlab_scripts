import argparse
import numpy as np
from Compute_trajectory_length import get_trajectory_distance


def get_height_diff(traj_file):
    with open(filename) as traj_file:
        lines = traj_file.readlines()

        num_of_points = int(lines[4].split()[1])

        lines = lines[5:]
        first_point = np.array([float(x) for x in lines[0].split()])
        last_point = np.array([float(x) for x in lines[num_of_points - 1].split()])

        print(first_point, last_point)
        difference = first_point[2] + last_point[2]
        print(f"Z difference is: {difference} m")
        return difference


def get_angle(filename):
    distance = get_trajectory_distance(filename)

    height_diff = get_height_diff(filename)

    angle = np.arctan2(height_diff, distance)
    print(
        f"angle is {np.rad2deg(angle)} deg, which is {angle} rad. Should be corrected by {-angle} in pitch, but it may depend on the IMU configuration")
    return angle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="Name of the trajectory file",
                        default="/home/mbo/data/warthog1_traj.vtk", type=str)

    args = parser.parse_args()
    filename = args.filename
    get_angle(filename)

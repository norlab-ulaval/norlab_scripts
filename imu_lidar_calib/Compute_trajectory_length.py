import argparse
import numpy as np


def get_trajectory_distance(filename):
    with open(filename) as traj_file:
        lines = traj_file.readlines()

        ctr = 0
        num_of_points = 0
        distance = 0
        for line in lines:
            ctr += 1
            if ctr < 5:
                continue
            elif ctr == 5:
                num_of_points = int(line.split()[1])
                continue
            if ctr > num_of_points - 1:
                break
            point = np.array([float(x) for x in lines[ctr].split()])
            next_point = np.array([float(x) for x in lines[ctr + 1].split()])

            distance += np.linalg.norm(point - next_point)
        print(f"Total distance is: {distance} m")
        traj_file.close()
        return distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="Name of the trajectory file", default="/home/mbo/data/warthog1_traj.vtk", type=str)

    args = parser.parse_args()
    filename = args.filename
    get_trajectory_distance(filename)

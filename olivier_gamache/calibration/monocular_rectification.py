import cv2
import yaml
import numpy as np
import os
from tqdm import tqdm

def main():
    ########### PARAMS ############ 
    EXPERIMENT = "campus-08-14-2024"
    CALIB_NAME = "backpack_2024-08-14-11-21-12"
    WHICH_CAMERA = "right"
    RUN = "backpack_2024-08-17-15-21-18"
    
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    CALIB_FILE = os.path.join(SCRIPT_DIR, '..', '..', '..','data', EXPERIMENT, 'data_high_resolution', CALIB_NAME, f'{WHICH_CAMERA}_mono_calib.yaml')
    IMAGES_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'data', EXPERIMENT, 'data_high_resolution', RUN, f'camera_{WHICH_CAMERA}')
    SAVE_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'data', EXPERIMENT, 'data_high_resolution', RUN, f'camera_{WHICH_CAMERA}_rectify')
    ###############################
    
    # Get the camera calibration parameters
    calib = yaml.safe_load(open(CALIB_FILE, 'r'))
    camera_matrix = np.array(calib['camera_matrix']['data']).reshape((3, 3))
    distortion_coefficients = np.array(calib['distortion_coefficients']['data'])
    
    # Create the save directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for file in tqdm(sorted(os.listdir(IMAGES_DIR))):
        if file.endswith('.png'):
            img = cv2.imread(os.path.join(IMAGES_DIR, file), cv2.IMREAD_ANYDEPTH)
            
            img_undistorted = cv2.undistort(img, camera_matrix, distortion_coefficients)
            cv2.imwrite(os.path.join(SAVE_DIR, file), img_undistorted)

if __name__ == "__main__":
    main()
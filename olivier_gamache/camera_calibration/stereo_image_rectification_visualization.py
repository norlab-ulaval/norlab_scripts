import cv2
import yaml
import numpy as np
import os

def display_rectification(img_left, img_right, stereo_map_left, stereo_map_right, shape):

    img_left_rect = cv2.remap(img_left, stereo_map_left[0], stereo_map_left[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    img_right_rect = cv2.remap(img_right, stereo_map_right[0], stereo_map_right[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
    img_left_rect_color = cv2.cvtColor(img_left_rect, cv2.COLOR_GRAY2BGR)
    img_right_rect_color = cv2.cvtColor(img_right_rect, cv2.COLOR_GRAY2BGR)
    
    for i in range(0, shape[1], 100):
        cv2.line(img_left_rect_color, (0, i), (shape[0], i), (0, 0, 255), 1)
        cv2.line(img_right_rect_color, (0, i), (shape[0], i), (0, 0, 255), 1)

    combined_image = np.hstack([img_left, img_right])
    combined_image_rect = np.hstack([img_left_rect_color, img_right_rect_color])
    combined_image_final = np.vstack([combined_image, combined_image_rect])
    combined_image_final = cv2.resize(combined_image_final, (0,0), fx=0.4, fy=0.4)
    cv2.imshow('Side by Side Images', combined_image_final)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return


def main():
    ########### PARAMS ############ 
    CALIB_NAME = "BOREALHDR_april.yaml"
    ACQUISITION_DAY = 'forest-04-21-2023'
    EXPERIMENT = 'backpack_2023-04-21-08-51-27'
    BOOL_12_BITS = True
    
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    CALIB_FILE = os.path.join(SCRIPT_DIR, 'calib_files_python', CALIB_NAME)
    IMAGES_DIR_LEFT = os.path.join(SCRIPT_DIR, '..', '..', '..', 'data', ACQUISITION_DAY, 'data_high_resolution', EXPERIMENT, 'camera_left','8.0')
    IMAGES_DIR_RIGHT = os.path.join(SCRIPT_DIR, '..', '..', '..', 'data', ACQUISITION_DAY, 'data_high_resolution', EXPERIMENT, 'camera_right','8.0')
    ###############################
    
    
    calib = yaml.safe_load(open(CALIB_FILE, 'r'))

    K_left = np.array(calib['LEFT.K']['data']).reshape((3, 3))
    K_right = np.array(calib['RIGHT.K']['data']).reshape((3, 3))
    distL = np.array(calib['LEFT.D']['data'])
    distR = np.array(calib['RIGHT.D']['data'])
    rectL = np.array(calib['LEFT.R']['data']).reshape((3, 3))
    rectR = np.array(calib['RIGHT.R']['data']).reshape((3, 3))
    P_left = np.array(calib['LEFT.P']['data']).reshape((3, 4))
    P_right = np.array(calib['RIGHT.P']['data']).reshape((3, 4))
    shape = (calib['LEFT.width'], calib['LEFT.height'])

    stereo_map_left = cv2.initUndistortRectifyMap(K_left, distL, rectL, P_left, shape, cv2.CV_16SC2)
    stereo_map_right = cv2.initUndistortRectifyMap(K_right, distR, rectR, P_right, shape, cv2.CV_16SC2)

    for file in sorted(os.listdir(IMAGES_DIR_LEFT)):
        if file.endswith('.png'):
            img_left = cv2.imread(os.path.join(IMAGES_DIR_LEFT, file), cv2.IMREAD_ANYDEPTH)
            img_right = cv2.imread(os.path.join(IMAGES_DIR_RIGHT, file), cv2.IMREAD_ANYDEPTH)

            if BOOL_12_BITS:
                img_left = (img_left/16.0).astype(np.uint8)
                img_right = (img_right/16.0).astype(np.uint8)
            display_rectification(img_left, img_right, stereo_map_left, stereo_map_right, shape)

if __name__ == "__main__":
    main()
import cv2
import os
import numpy as np

class ORB_FEATURES:
    def __init__(self, orb_parameters):
        self.orb_parameters = orb_parameters
    
    def find_orb_features(self, image):
        # Convert the image to grayscale
        gray = image
        
        # Create an ORB object
        orb = cv2.ORB_create(nfeatures=self.orb_parameters["nfeatures"], 
                             scaleFactor=self.orb_parameters["scaleFactor"], 
                             nlevels=self.orb_parameters["nlevels"], 
                             edgeThreshold=self.orb_parameters["edgeThreshold"], 
                             firstLevel=self.orb_parameters["firstLevel"], 
                             WTA_K=self.orb_parameters["WTA_K"], 
                             scoreType=self.orb_parameters["scoreType"], 
                             patchSize=self.orb_parameters["patchSize"], 
                             fastThreshold=self.orb_parameters["fastThreshold"])
        
        # Detect and compute the ORB features
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def calculate_matches(self, descriptors_1, descriptors_2, best_k=10):
        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        # Match the keypoints
        matches = bf.match(descriptors_1, descriptors_2)
        
        # Sort the matches based on distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Get the best matches
        matches = matches[:best_k]
        return matches

class SIFT_FEATURES:
    def __init__(self, sift_parameters):
        self.sift_parameters = sift_parameters
    
    def find_sift_features(self, image):
        # Convert the image to grayscale
        gray = image
        
        # Create an SIFT object
        sift = cv2.SIFT_create(nfeatures=self.sift_parameters["nfeatures"], 
                               nOctaveLayers=self.sift_parameters["nOctaveLayers"], 
                               contrastThreshold=self.sift_parameters["contrastThreshold"], 
                               edgeThreshold=self.sift_parameters["edgeThreshold"], 
                               sigma=self.sift_parameters["sigma"])
        
        # Detect and compute the SIFT features
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def calculate_matches(self, descriptors_1, descriptors_2, best_k=10):
        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        # Match the keypoints
        matches = bf.match(descriptors_1, descriptors_2)
        
        # Sort the matches based on distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Get the best matches
        matches = matches[:best_k]
        return matches

class VISUALIZER:
    def __init__(self):
        self.image_1_path = None
        self.image_2_path = None
        self.image_1 = None
        self.image_2 = None
        self.width = None
        self.height = None
        
    def load_image(self, image_1_path, image_2_path):
        # Load the image
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path
        image_1 = cv2.imread(self.image_1_path, cv2.IMREAD_ANYDEPTH)
        self.image_1 = self.convert_to_8bits(image_1)
        image_2 = cv2.imread(self.image_2_path, cv2.IMREAD_ANYDEPTH)
        self.image_2 = self.convert_to_8bits(image_2)
        self.height, self.width = self.image_1.shape
    
    def convert_to_8bits(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
        image_8bits = (image_gray/2**4).astype('uint8')
        return image_8bits
        
    def draw_matches_on_image(self, keypoints_1, keypoints_2, matches):
        # Draw the matches
        img_matches = cv2.drawMatches(self.image_1, keypoints_1, self.image_2, keypoints_2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_matches
    
    def display_image(self, image):
        img_to_display = cv2.resize(image, None, fx=0.4, fy=0.4)
        cv2.imshow("Image", img_to_display)
        print("Press any key to close the window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save_image(self, image, save_path):
        # Save the image
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)
    
def main():
    # Path to the images
    IMAGES_1 = [
        "/media/alienware/T7_Shield/ICRA2024_OG/dataset/belair-09-27-2023/data_high_resolution/backpack_2023-09-27-13-20-03/camera_left/8.0/1695835261846207488.png",
        
    ]
    IMAGES_2 = [
        "/media/alienware/T7_Shield/ICRA2024_OG/dataset/belair-09-27-2023/data_high_resolution/backpack_2023-09-27-13-20-03/camera_left/8.0/1695835262119823104.png",
    ]
    WHICH_FEATURE = "sift" # "sift" or "orb"

    print("\033[93mWarning: To have accurate matches, should use rectified images\033[0m")
    
    orb_params = {
        "nfeatures": 6000,
        "scaleFactor": 1.2,
        "nlevels": 16,
        "edgeThreshold": 5,
        "firstLevel": 0,
        "WTA_K": 2,
        "scoreType": cv2.ORB_HARRIS_SCORE,
        "patchSize": 5,
        "fastThreshold": 14,
    }
    
    sift_params = {
        "nfeatures": 6000,
        "nOctaveLayers": 3,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6,
    }

    if WHICH_FEATURE == "sift":
        feature_extractor = SIFT_FEATURES(sift_params)
        for image_1_path, image_2_path in zip(IMAGES_1, IMAGES_2):
            visualizer = VISUALIZER()
            visualizer.load_image(image_1_path, image_2_path)
            visualizer.save_image(visualizer.image_1, "images/matches/original_" + image_1_path.split("/")[-1])
            visualizer.save_image(visualizer.image_2, "images/matches/original_" + image_2_path.split("/")[-1])
            keypoints_1, descriptors_1 = feature_extractor.find_sift_features(visualizer.image_1)
            keypoints_2, descriptors_2 = feature_extractor.find_sift_features(visualizer.image_2)
            matches = feature_extractor.calculate_matches(descriptors_1, descriptors_2, best_k=100)
            img_matches = visualizer.draw_matches_on_image(keypoints_1, keypoints_2, matches)
            visualizer.display_image(img_matches)
            visualizer.save_image(img_matches, "images/matches/sift_" + image_1_path.split("/")[-1] + "_" + image_2_path.split("/")[-1])

    elif WHICH_FEATURE == "orb":
        feature_extractor = ORB_FEATURES(orb_params)
        for image_1_path, image_2_path in zip(IMAGES_1, IMAGES_2):
            visualizer = VISUALIZER()
            visualizer.load_image(image_1_path, image_2_path)
            keypoints_1, descriptors_1 = feature_extractor.find_orb_features(visualizer.image_1)
            keypoints_2, descriptors_2 = feature_extractor.find_orb_features(visualizer.image_2)
            matches = feature_extractor.calculate_matches(descriptors_1, descriptors_2, best_k=100)
            img_matches = visualizer.draw_matches_on_image(keypoints_1, keypoints_2, matches)
            visualizer.display_image(img_matches)
            visualizer.save_image(img_matches, "images/matches/orb_" + image_1_path.split("/")[-1] + "_" + image_2_path.split("/")[-1])


if __name__ == "__main__":
    main()
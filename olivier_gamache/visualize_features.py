import cv2
import os

class ORB_FEATURES:
    def __init__(self, orb_parameters):
        self.orb_parameters = orb_parameters
    
    def find_orb_features(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
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
        return keypoints

class SIFT_FEATURES:
    def __init__(self, sift_parameters):
        self.sift_parameters = sift_parameters
    
    def find_sift_features(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create an SIFT object
        sift = cv2.SIFT_create(nfeatures=self.sift_parameters["nfeatures"], 
                               nOctaveLayers=self.sift_parameters["nOctaveLayers"], 
                               contrastThreshold=self.sift_parameters["contrastThreshold"], 
                               edgeThreshold=self.sift_parameters["edgeThreshold"], 
                               sigma=self.sift_parameters["sigma"])
        
        # Detect and compute the SIFT features
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints

class VISUALIZER:
    def __init__(self):
        self.image_path = None
        self.image = None
        self.width = None
        self.height = None
        
    def load_image(self, image_path):
        # Load the image
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.height, self.width, _ = self.image.shape
        
    def draw_keypoints_on_image(self, keypoints):
        # Draw the keypoints
        image_with_keypoints = cv2.drawKeypoints(self.image, keypoints, None, color=(0, 255, 0), flags=0)
        return image_with_keypoints
    
    def display_image(self, image):
        img_to_display = cv2.resize(image, None, fx=0.5, fy=0.5)
        cv2.imshow("Image", img_to_display)
        print("Press any key to close the window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_grid_on_image(self, image, grid_size=20, keypoints=None):
        # Draw the grid on the image
        step_height = self.height // grid_size
        step_width = self.width // grid_size
        for i in range(0, self.height, step_height):
            cv2.line(image, (0, i), (self.width, i), (0, 0, 0), 2)
        for i in range(0, self.width, step_width):
            cv2.line(image, (i, 0), (i, self.height), (0, 0, 0), 2)

        # Colorize the grid based on the presence of keypoints
        if keypoints is not None:
            image = self.display_image_with_colorized_grid = self.display_image_with_colorized_grid(image, grid_size, keypoints)
        return image
    
    def display_image_with_colorized_grid(self, image, grid_size, keypoints):
        step_height = self.height // grid_size
        step_width = self.width // grid_size
        for i in range(0, self.height, step_height):
            cv2.line(image, (0, i), (self.width, i), (0, 0, 0), 2)
            for j in range(0, self.width, step_width):
                cell_keypoints = [kp for kp in keypoints if i <= kp.pt[1] < i + step_height and j <= kp.pt[0] < j + step_width]
                overlay = image.copy()
                if cell_keypoints:
                    cv2.rectangle(overlay, (j, i), (j + step_width, i + step_height), (0, 255, 0), -1)
                else:
                    cv2.rectangle(overlay, (j, i), (j + step_width, i + step_height), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        return image
    
    def save_image(self, image, save_path):
        # Save the image
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)
    
def main():
    # Path to the images
    IMAGES = [
        "/media/alienware/T7_Shield/video_iros2024/IROS2024/figures/imgs_result/ae-50/00108.png",
    ]
    WHICH_FEATURE = "sift" # "sift" or "orb"
    
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

    visualizer = VISUALIZER()
    if WHICH_FEATURE == "sift":
        sift_features = SIFT_FEATURES(sift_params)
        for image_path in IMAGES:
            visualizer.load_image(image_path)
            visualizer.save_image(visualizer.image, "images/features/original_" + image_path.split("/")[-1])

            keypoints = sift_features.find_sift_features(visualizer.image)
            image_with_keypoints = visualizer.draw_keypoints_on_image(keypoints)
            visualizer.display_image(image_with_keypoints)
            visualizer.save_image(image_with_keypoints, "images/features/sift_" + image_path.split("/")[-1])

            image_with_grid = visualizer.display_grid_on_image(image_with_keypoints, grid_size=20)
            visualizer.display_image(image_with_grid)
            visualizer.save_image(image_with_grid, "images/features/sift_grid_" + image_path.split("/")[-1])

            image_colorized_grid = visualizer.display_grid_on_image(image_with_keypoints, 20, keypoints)
            visualizer.display_image(image_colorized_grid)
            visualizer.save_image(image_colorized_grid, "images/features/sift_colorized_grid_" + image_path.split("/")[-1])

    elif WHICH_FEATURE == "orb":
        orb_features = ORB_FEATURES(orb_params)
        for image_path in IMAGES:
            visualizer.load_image(image_path)
            visualizer.save_image(visualizer.image, "images/features/original_" + image_path.split("/")[-1])

            keypoints = orb_features.find_orb_features(visualizer.image)
            image_with_keypoints = visualizer.draw_keypoints_on_image(keypoints)
            visualizer.display_image(image_with_keypoints)
            visualizer.save_image(image_with_keypoints, "images/features/orb_" + image_path.split("/")[-1])

            image_with_grid = visualizer.display_grid_on_image(image_with_keypoints, grid_size=20)
            visualizer.display_image(image_with_grid)
            visualizer.save_image(image_with_grid, "images/features/orb_grid_" + image_path.split("/")[-1])

            image_colorized_grid = visualizer.display_grid_on_image(image_with_keypoints, 20, keypoints)
            visualizer.display_image(image_colorized_grid)
            visualizer.save_image(image_colorized_grid, "images/features/orb_colorized_grid_" + image_path.split("/")[-1])

if __name__ == "__main__":
    main()
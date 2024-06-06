import cv2

class ORB_VISUALIZER:
    def __init__(self, orb_parameters):
        self.orb_parameters = orb_parameters
        self.image_path = None
        self.image = None
        self.image_with_keypoints = None
        self.keypoints = None
        
    def load_image(self, image_path):
        # Load the image
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
    
    def find_orb_features(self):
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
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
        self.keypoints, descriptors = orb.detectAndCompute(gray, None)
        
    def draw_keypoints(self):
        # Draw the keypoints
        self.image_with_keypoints = cv2.drawKeypoints(self.image, self.keypoints, None, color=(0, 255, 0), flags=0)
    
    def display_image_with_keypoints(self):
        # Display the image
        cv2.imshow("ORB Features", self.image_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save_image_with_keypoints(self, save_path):
        # Save the image
        cv2.imwrite(save_path, self.image_with_keypoints)
    
def main():
    # Path to the images
    images = [
              "/home/olivier_g/Documents/ICRA2024/ICRA2024_Olivier_Gamache_Publication/Figures/introduction/1ms.png",
              "/home/olivier_g/Documents/ICRA2024/ICRA2024_Olivier_Gamache_Publication/Figures/introduction/2ms.png",
              "/home/olivier_g/Documents/ICRA2024/ICRA2024_Olivier_Gamache_Publication/Figures/introduction/16ms.png",
              "/home/olivier_g/Documents/ICRA2024/ICRA2024_Olivier_Gamache_Publication/Figures/introduction/32ms.png",
    ]
    
    orb_params = {
        "nfeatures": 3000,
        "scaleFactor": 1.2,
        "nlevels": 8,
        "edgeThreshold": 5,
        "firstLevel": 0,
        "WTA_K": 2,
        "scoreType": cv2.ORB_HARRIS_SCORE,
        "patchSize": 5,
        "fastThreshold": 14,
    }
    
    orb_visualizer = ORB_VISUALIZER(orb_params)
    for image in images:
        orb_visualizer.load_image(image)
        orb_visualizer.find_orb_features()
        orb_visualizer.draw_keypoints()
        orb_visualizer.display_image_with_keypoints()
        orb_visualizer.save_image_with_keypoints(image.replace(".png", "_orb.png"))

if __name__ == "__main__":
    main()
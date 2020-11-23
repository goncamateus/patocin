import os
os.sys.path.append("../../perception")

from Perception import Perception
import cv2
import numpy as np

class AdvancedLaneDetection(Perception):
    def __init__(self, points_src, points_dst, yellow_threshold, white_threshold):
        self.points_src = points_src
        self.points_dst = points_dst
        self.yellow_threshold = yellow_threshold
        self.white_threshold = white_threshold

    def Perceive(self, data):
        assert data.shape == (480, 640, 3), "AdvancedLaneDetector must receive a (480, 640, 3) image"

        # 1. Warp the image to a bird's eye view
        M = self.getTransform(self.points_src, self.points_dst)
        warped_image = cv2.warpPerspective(data, M, (data.shape[0], data.shape[1]), flags=cv2.INTER_AREA)
        
        # 2. Change the color space of the image from RGB to HSV
        hsv_image = self.toHSV(warped_image)

        return hsv_image 

    def getTransform(self, points_src, points_dst):
        """
            Function that compute the transformation between two sets of points.
        """
        
        src = np.float32(points_src)
        dst = np.float32(points_dst)
    
        M = cv2.getPerspectiveTransform(src, dst)

        return M
    
    def toHSV(self, image):
        """
            Function to transform the color space to HSV
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
import os
os.sys.path.append("../../perception")

from Perception import Perception
import cv2

class AdvancedLaneDetection(Perception):
    def __init__(self, yellow_threshold, white_threshold):
        self.yellow_threshold = yellow_threshold
        self.white_threshold = white_threshold

    def Perceive(self, data):
        assert data.shape == (480, 640, 3), "AdvancedLaneDetector must receive a (480, 640, 3) image"

        data = self.toHSV(data)
        return data 

    def toHSV(self, image):
        """
            Function to transform the color space to HSV
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
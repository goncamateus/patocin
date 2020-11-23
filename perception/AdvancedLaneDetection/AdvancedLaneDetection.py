import os
os.sys.path.append("../../perception")

from Perception import Perception

class AdvancedLaneDetection(Perception):
    def __init__(self, yellow_threshold, white_threshold):
        self.yellow_threshold = yellow_threshold
        self.white_threshold = white_threshold

    def Perceive(self, data):
        assert data.shape == (480, 640, 3), "AdvancedLaneDetector must receive a (480, 640, 3) image"

        return data        
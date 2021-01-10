import os
os.sys.path.append("../../perception")

from Perception import Perception
import cv2
import numpy as np

class AdvancedLaneDetection(Perception):
    def __init__(self, points_src, points_dst, yellow_threshold, white_threshold, min_pix=100):
        self.points_src = points_src
        self.points_dst = points_dst
        self.yellow_threshold = yellow_threshold
        self.white_threshold = white_threshold
        self.min_pix = min_pix

    def Perceive(self, data):
        assert data.shape == (480, 640, 3), "AdvancedLaneDetector must receive a (480, 640, 3) image"

        # 1. Warp the image to a bird's eye view
        M = self.getTransform()
        warped_image = cv2.warpPerspective(data, M, (data.shape[0], data.shape[1]), flags=cv2.INTER_AREA)
        
        # 2. Change the color space of the image from RGB to HSV
        hsv_image = self.toHSV(warped_image)

        # 3. Segment the lanes by their color (yellow and white)
        yellow_thresh, white_threshold = self.segmentLanes(hsv_image)
        binary = cv2.bitwise_or(yellow_thresh, white_threshold)

        # 4. Find left and right lanes
        w_height = 30
        w_width = 40

        yellow_lane_dict = self.findLane(yellow_thresh, w_height, w_width, name="Yellow", debug=True)
        white_lane_dict = self.findLane(white_threshold, w_height, w_width, name="White", debug=True)
        #white_lane_dict = findLane(white_lane, w_height, w_width, min_pix, debug)

        #yellow_hist = np.sum(yellow_thresh, axis=0)
        
        return binary, yellow_lane_dict, white_lane_dict  

    def segmentLanes(self, image):
        """
            Function that applies the segmentation thresholds to the HSV image.
        """
        yellow = cv2.inRange(image, self.yellow_threshold[0], self.yellow_threshold[1])
        white = cv2.inRange(image, self.white_threshold[0], self.white_threshold[1])

        return yellow, white

    def getTransform(self):
        """
            Function that compute the transformation between two sets of points.
        """
        
        src = np.float32(self.points_src)
        dst = np.float32(self.points_dst)
    
        M = cv2.getPerspectiveTransform(src, dst)

        return M
    
    def toHSV(self, image):
        """
            Function to transform the color space to HSV
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def findLane(self, binary_image, w_height, w_width, name, debug=False):
    
        dict_find_lanes = {}
        
        binary_hist = np.sum(binary_image, axis=0)
        nonzero = binary_image.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        w_x = int(np.argmax(binary_hist))
        w_y = int(binary_image.shape[0] - w_height/2)

        #find first window
        dict_first_window = self.findFirstWindow(w_x, w_y, w_height, w_width, nonzero_x, nonzero_y, debug)

        if dict_first_window["found"]:
            dict_remaining_windows = self.findRemainingWindows(dict_first_window["w_center"], w_height, w_width, nonzero_x, nonzero_y, name, debug)
            
            dict_find_lanes["windows"] = [dict_first_window["w_center"]] + dict_remaining_windows["chosen_windows"]
            dict_find_lanes["found"] = True

            if debug:
                initial_window_img = self.debugInitialWindow(binary_image, dict_first_window["w_centroids"], w_width, w_height)
                remaining_windows_img = self.debugRemainingWindows(binary_image, dict_remaining_windows, w_height, w_width)
                
                dict_find_lanes["debug_images"] = [initial_window_img, remaining_windows_img]
        else:
            print(name + " lane didn't found")
            dict_find_lanes["found"] = False
            
        return dict_find_lanes

    def findRemainingWindows(self, w_center, w_height, w_width, nonzero_x, nonzero_y, name, debug=False):
        offsets = [(-w_width, 0), (-w_width, -w_height), (0, -w_height), (w_width, w_height), (w_width, 0)] #left, upper-left, above, uper right, right
        
        dict_remaining_w = {}
        
        if debug:
            dict_remaining_w["search_windows"] = []
        dict_remaining_w["chosen_windows"] = []
        
        current_x = w_center[0]
        current_y = w_center[1]
        
        done = False
        count = 0
        while not done and (count < 30):
            best_offset_index = -1
            best_offset_p = -1
            best_inside_indexes = None

            for i, off in enumerate(offsets):
                w_x = current_x + off[0]
                w_y = current_y + off[1]

                w_x_high = int(w_x - w_width/2)
                w_y_high = int(w_y - w_height/2)

                w_x_low = int(w_x + w_width/2)
                w_y_low = int(w_y + w_height/2)

                if w_x_high < 0 or w_x_low > 480 or w_y_high < 0 or w_y_low > 640:
                    pixels_inside = 0
                    inside_window = None
                else:
                    inside_window = ( (nonzero_y >= w_y_high) & (nonzero_y < w_y_low) & 
                (nonzero_x >= w_x_high) &  (nonzero_x <= w_x_low)).nonzero()[0]
                    pixels_inside = len(inside_window)

                    if debug:
                        dict_remaining_w["search_windows"].append((w_x,w_y))
                
                if pixels_inside > best_offset_p:
                    best_offset_index = i
                    best_offset_p = pixels_inside
                    best_inside_indexes = inside_window

            if best_offset_p <= 0:
                if name == "Yellow":
                    #Gaps in yellow line can be bigger than the size of the box so try one last time
                    w_x = current_x + 0
                    w_y = current_y + -w_height*1.5

                    w_x_high = int(w_x - w_width/2)
                    w_y_high = int(w_y - w_height/2)

                    w_x_low = int(w_x + w_width/2)
                    w_y_low = int(w_y + w_height/2)

                    if w_x_high < 0 or w_x_low > 480 or w_y_high < 0 or w_y_low > 640:
                        pixels_inside = 0
                        inside_window = None
                    else:
                        inside_window = ( (nonzero_y >= w_y_high) & (nonzero_y < w_y_low) & 
                    (nonzero_x >= w_x_high) &  (nonzero_x <= w_x_low)).nonzero()[0]
                        pixels_inside = len(inside_window)

                        if debug:
                            dict_remaining_w["search_windows"].append((w_x,w_y))
                    
                    if pixels_inside > best_offset_p:
                        best_offset_index = i
                        best_offset_p = pixels_inside
                        best_inside_indexes = inside_window
                    
                    if best_offset_p <= 0:
                        done = True
                    else:
                        current_x = int(np.mean(nonzero_x[best_inside_indexes]))
                        current_y = int(np.mean(nonzero_y[best_inside_indexes]))
                        if len(dict_remaining_w["chosen_windows"]) > 0 and current_x == dict_remaining_w["chosen_windows"][-1][0] and current_y == dict_remaining_w["chosen_windows"][-1][1]:
                            done = True
                        dict_remaining_w["chosen_windows"].append((current_x,current_y))
                else:
                    done = True
            else:
                current_x = int(np.mean(nonzero_x[best_inside_indexes]))
                current_y = int(np.mean(nonzero_y[best_inside_indexes]))
                if len(dict_remaining_w["chosen_windows"]) > 0 and current_x == dict_remaining_w["chosen_windows"][-1][0] and current_y == dict_remaining_w["chosen_windows"][-1][1]:
                    done = True
                #print(current_x, current_y)
                dict_remaining_w["chosen_windows"].append((current_x,current_y))
            
            count += 1
        #print(done, count)
        return dict_remaining_w

    def findFirstWindow(self, w_x, w_y, w_height, w_width, nonzero_x, nonzero_y, debug=False):
    
        dict_first_window = {}
        if debug:
            dict_first_window["w_centroids"] = []
        
        done = False
        dict_first_window["found"] = False
        
        while not done:
            if debug:
                dict_first_window["w_centroids"].append([w_x,w_y])

            w_x_high = int(w_x - w_width/2)
            w_y_high = int(w_y - w_height/2)

            w_x_low = int(w_x + w_width/2)
            w_y_low = int(w_y + w_height/2)

            if w_x_high < 0 or w_x_low > 480 or w_y_high < 0 or w_y_low > 640:
                done = True
                continue
            
            inside_window = ( (nonzero_y >= w_y_high) & (nonzero_y < w_y_low) & 
                (nonzero_x >= w_x_high) &  (nonzero_x <= w_x_low)).nonzero()[0]

            if len(inside_window) > self.min_pix:
                w_x = int(np.mean(nonzero_x[inside_window]))
                w_y = int(np.mean(nonzero_y[inside_window]))
                
                if debug:
                    dict_first_window["w_centroids"].append([w_x, w_y])

                done = True
                dict_first_window["found"] = True
                dict_first_window["w_center"] = (w_x, w_y)
            else:
                w_y = int(w_y - w_height)
                
        return dict_first_window

    def debugRemainingWindows(self, img, dict_remaining_windows, w_height, w_width):
        remaining_windows_img = np.dstack((img, img, img))

        for window in dict_remaining_windows["search_windows"]:
            w_x_high = int(window[0] - w_width/2)
            w_y_high = int(window[1] - w_height/2)

            w_x_low = int(window[0] + w_width/2)
            w_y_low = int(window[1] + w_height/2)

            remaining_windows_img = cv2.rectangle(remaining_windows_img, (w_x_high, w_y_high), (w_x_low, w_y_low), (255,0,0), 5)

        for window in dict_remaining_windows["chosen_windows"]:
            w_x_high = int(window[0] - w_width/2)
            w_y_high = int(window[1] - w_height/2)

            w_x_low = int(window[0] + w_width/2)
            w_y_low = int(window[1] + w_height/2)

            remaining_windows_img = cv2.rectangle(remaining_windows_img, (w_x_high, w_y_high), (w_x_low, w_y_low), (255,255,0), 5)

        return remaining_windows_img

    def debugInitialWindow(self, image, initial_centroids, w_width, w_height):
        initial_window_img = np.dstack((image, image, image))

        for window in initial_centroids[:-1]:
            w_x_high = int(window[0] - w_width/2)
            w_y_high = int(window[1] - w_height/2)

            w_x_low = int(window[0] + w_width/2)
            w_y_low = int(window[1] + w_height/2)

            initial_window_img = cv2.rectangle(initial_window_img, (w_x_high, w_y_high), (w_x_low, w_y_low), (255,0,0), 5)

        window = initial_centroids[-1]
        w_x_high = int(window[0] - w_width/2)
        w_y_high = int(window[1] - w_height/2)

        w_x_low = int(window[0] + w_width/2)
        w_y_low = int(window[1] + w_height/2)

        initial_window_img = cv2.rectangle(initial_window_img, (w_x_high, w_y_high), (w_x_low, w_y_low), (255,255,0), 5)
        
        return initial_window_img  
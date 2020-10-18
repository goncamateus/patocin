from perception import Perception
import numpy as np
import cv2
from scipy import stats

class NaiveLaneDetection(Perception):
    def __init__(self, vertices, g_kernel, canny_low_threshold, canny_high_threshold, hough_rho, hough_theta, hough_threshold, hough_min_line_len, hough_max_line_gap):
        self.vertices = vertices
        self.gaussian_kernel = g_kernel
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap

    def Perceive(self, data):
        assert data.shape == (480,640,3), "NaiveLaneDetector must receive a (480,640,3) image"
        gray = self.GrayScale(data)
        g_blur = self.GaussianBlur(gray, self.gaussian_kernel)
        canny = self.Canny(g_blur, self.canny_low_threshold, self.canny_high_threshold)
        roi = self.RegionOfInterest(canny, self.vertices)
        line_image, lines = self.HoughLines(roi, self.hough_rho, self.hough_theta, self.hough_threshold, self.hough_min_line_len, self.hough_max_line_gap)
        if lines is not None:
            left_lines, right_lines, separed_lines_image = self.SeparateLines(line_image, lines)
            final, left_line, right_line = self.AverageLines(left_lines, right_lines, separed_lines_image)
            return separed_lines_image, final, left_line, right_line
        else:
            return line_image, line_image, (None, None), (None, None)
        #final = self.AverageLines(left_lines, right_lines, separed_lines_image)

    def GrayScale(self, img):
        """Function that transforms the color input image in a grayscale one"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def GaussianBlur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def Canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def RegionOfInterest(self, img, vertices):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """

        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    def DrawLines(self, img, lines, color=[255, 0, 0], thickness=2):
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def HoughLines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        img should be the output of a Canny transform.
        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        if lines is not None:
            self.DrawLines(line_img, lines)
        else:
            print("No lines detected")
        return line_img, lines
    
    def SeparateLines(self, image, lines):
        """
            Separate lines into left and right by their slope
        """
        separed_lines = np.zeros_like(image)
        
        left= []
        right = []
        
        for line in lines:
            for x1,y1,x2,y2 in line:
                dy = y2 - y1
                dx = x2 - x1
                
                if(dy == 0 or dx == 0):
                    continue
                
                A = dy/dx
                if(abs(A) < 0.2):
                    continue

                if(A < 0):
                    left.append(line)
                    cv2.line(separed_lines, (x1, y1), (x2, y2), [255,0,0], 2)
                else:
                    right.append(line)
                    cv2.line(separed_lines, (x1, y1), (x2, y2), [255,255,0], 2)
        
        if len(left) < 4:
            left = []
        if len(right) < 4:
            right = []
        return left, right, separed_lines

    def AverageLines(self, left_lines, right_lines, image):
        line_image = np.zeros_like(image)
        
        left_line_p1 = None
        left_line_p2 = None
        right_line_p1 = None
        right_line_p2 = None

        if len(left_lines) > 0:
            a = []
            b = []

            for line in left_lines:
                for x1,y1,x2,y2 in line:
                    a_aux = (y2-y1) / (x2-x1)
                    b_aux = y2 - a_aux*x2

                    a.append(a_aux)
                    b.append(b_aux)
            #Find the line(y = Ax + B) that best fit the points    
            #left_A, left_B, _, _, _ = stats.linregress(pts_x, pts_y)
            left_A = np.array(a).mean()
            left_B = np.array(b).mean()

            left_line_p1 = (0, int(left_B))
            left_line_p2 = (int((150-left_B)/left_A), 150)
            cv2.line(line_image, left_line_p1, left_line_p2, [255,0,0], 10)

        if len(right_lines) > 0:
            a = []
            b = []

            for line in right_lines:
                for x1,y1,x2,y2 in line:
                    a_aux = (y2-y1) / (x2-x1)
                    b_aux = y2 - a_aux*x2

                    a.append(a_aux)
                    b.append(b_aux)
            #Find the line(y = Ax + B) that best fit the points    
            #left_A, left_B, _, _, _ = stats.linregress(pts_x, pts_y)
            right_A = np.array(a).mean()
            right_B = np.array(b).mean()

            right_line_p1 = (639, int(right_A*639 + right_B))
            right_line_p2 = (int((150-right_B)/right_A), 150)
            cv2.line(line_image, right_line_p1, right_line_p2, [255,255,0], 10)

        return line_image, (left_line_p1, left_line_p2), (right_line_p1, right_line_p2)
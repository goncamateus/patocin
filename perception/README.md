# Perception
 
## Table of Contents

1. [Introduction](#Introduction)
2. [Naive Lane Detection](#Naive-Lane-Detection)
3. [Advanced Lane Detection](#Advanced-Lane-Detection)

## Introduction
Before chosing actions, autonomous vehicles need to perceive its surrounding environment. Thi folder contrains a collection of perception methods to help the duckiebot perceive its surroundings. The Perception.py script defines the Perception object interface, it receives raw input observations and outputs usefull perception information as lane detection, semantic segmented images, object detection, traffic lights detection, etc.


## Naive Lane Detection

<p align="center">
<img src="https://github.com/goncamateus/patocin/blob/perception/perception/NaiveLaneDetection/NaiveLaneDetection.gif" width="500px"><br>
</p>

The Naive Lane Detection method employed here is probabily the simpler algorithm we could think for detecting lane lines. It is composed of a seven-step pipeline using classical computer vision processing:

1. Convert the color image to grayscale.
2. Apply a gaussian blur.
3. Find edges using the Canny edge detector.
4. Select the region of interest in the image.
5. Find lines using Hough transform.
6. Separate left and right lines using its slope.
7. Average left and right lines to obtain the estimative of left and right lanes.

This method can be tested using manual control:
```
cd NaiveLaneDetection
python ManualContolNaiveLaneDetection.py
```

It can also be tested autonomously, with a simple controller:
```
cd NaiveLaneDetection
python TestNaiveLaneDetection.py
```

Advantages
* Simple and easy to understand and code.
* As this methods cover only simple classical computer vision procesing, it outputs the requested information very fast compared to more complex methods. An important advantage as autonomous vehicles needs to take decisions in real time.

Drawbacks
* It has a lot of hand-tuned hyperparameters.
* It only covers very simple scenarios.
* As this method is designed to detect lines, it does well in the straight road scenario but fails in scenarios with curves. 

## Advanced Lane Detection

<p align="center">
<img src="https://github.com/goncamateus/patocin/blob/perception/perception/AdvancedLaneDetection/AdvancedLaneDetection.gif" width="500px"><br>
</p>

This method is more sofisticated than the previous one, but note that it stills use techniques from classic image processing. For detailed information, check out the step by step jupyter notebook in the Advanced lane detection folder. The Advanced Lane Detection consists in a five step pipeline as shown below: 

1. Warp the image to a bird's eye view.
2. Change the color space to HSV?
3. Threshold the lanes by the color (yellow and white).
4. Find lane beginning using histogram.
5. From the lane start position, use successive windows to find the entire lane.
6. Fit a third degree polynomial curve to the points inside the windows.

This method can be tested using manual control:
```
cd AdvancedLaneDetection
python ManualControlAdvancedLaneDetection.py
```

Advantages
* Not so complex and very fast, thus adpted to real time processing.
* Unlikely the Naive Lane Detection, it can detect curve lanes and be used to the Lane Follow challenge.

Drawbacks
* Still has a lot of hand-tuned hyperparameters
* Don't generalize to complex scenarios and struggles in sharp curves 
* As it is strongly based on a color threshold, it is not robust to light variation (can be easily fooled and misclassify lanes when there is yellow or white objects near the lanes)

Room for improvements:
* The lanes are assumed to have a minimal inclination, so very sharp curves are not well detected.
* Use gradient/edges information in addition to the color in order to detect the lanes.
* Use previous detected lanes to reduce the lane search in the next frame.
* Use previous detected lanes to remove false lane detections.
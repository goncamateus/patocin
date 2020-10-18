# Perception

## Table of Contents

1. [Introduction](#Introduction)
2. [Naive Lane Detection](#Naive-Lane-Detection)

## Introduction
Before chosing actions, autonomous vehicles need to perceive its surrounding environment. Thi folder contrains a collection of perception methods to help the duckiebot perceive its surroundings. The Perception.py script defines the Perception object interface, it receives raw input observations and outputs usefull perception information as lane detection, semantic segmented images, object detection, traffic lights detection, etc.


## Naive Lane Detection

The Naive Lane Detection method employed here is probabily the simpler algorithm we could think for detecting lane lines. It is composed of a seven-step pipeline using classical computer vision processing:

1. Convert the color image to grayscale
2. Apply a gaussian blur
3. Find edges using the Canny edge detector
4. Select the region of interest in the image
5. Find lines using Hough transform
6. Separate left and right lines using its slope
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
* It has a lot of hand-tuned hyperparameters
* It only covers very simple scenarios
* As this method is designed to detect lines, it does well in the straight road scenario but fails in scenarios with curves. 
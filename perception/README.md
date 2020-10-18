# Perception

## Table of Contents

1. [Introduction](#Introduction)
2. [Naive Lane Detection](#Naive-Lane-Detection)

## Introduction
Before chosing actions, autonomous vehicles need to perceive its surrounding environment. Thi folder contrains a collection of perception methods to help the duckiebot perceive its surroundings. The Perception.py script defines the Perception object interface, it receives raw input observations and outputs usefull perception information as lane detection, semantic segmented images, object detection, traffic lights detection, etc.


## Naive Lane Detection

The Naive Lane Detection method employed here is probabily the simpler algorithm we could think for detecting lane lines. It is composed of a pipeline of 7 steps using classical computer vision processing:

1. Convert the color image to grayscale
2. Apply a gaussian blur
3. Find edges using the Canny edge detector
4. Select the region of interest in the image
5. Find lines using Hough transform
6. Separate left and right lines using its slope
7. Avera left and right lines to obtain the estimative of left and right lanes.

Advantages
* As autonomous vehicles needs to take decisions in real time.

Drawbacks
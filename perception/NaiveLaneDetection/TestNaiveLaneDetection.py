import os
os.sys.path.append("../../gym-duckietown")

import numpy as np
import cv2
from gym_duckietown.envs.duckietown_env import *
from NaiveLaneDetection import NaiveLaneDetection


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def controler(left_line, right_line):
    """
        Simple controller to follow the lanes outputed by
        the Naive Lane Detector.
    """
    d_left = 0
    d_right = 0

    if left_line[0] is not None:
        x1 = left_line[0][0]
        y1 = left_line[0][1]
        x2 = left_line[1][0]
        y2 = left_line[1][1]
        d_left = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    if right_line[0] is not None:
        x1 = right_line[0][0]
        y1 = right_line[0][1]
        x2 = right_line[1][0]
        y2 = right_line[1][1]
        d_right = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    if d_left == 0 and d_right == 0:
        print("No lines detected")
        steer = 0
        throttle = 0
    else:
        throttle = 0.2
        diff = d_left - d_right
        steer = -diff/1000

    return steer, throttle

# Parameters
vertices = np.array([[(0,200), (640,200), (640,480), (0,480)]])
kernel = 5
low_threshold = 50
high_threshold = 150
rho = 1
theta = np.pi/180
threshold = 10
min_line_len = 10
max_line_gap = 10

NLD = NaiveLaneDetection(vertices, kernel, low_threshold, high_threshold, rho, theta, threshold, min_line_len, max_line_gap)
cv2.namedWindow("Lines")
cv2.namedWindow("Averaged Line")

env = DuckietownLF(map_name='straight_road',
                    max_steps=1500,
                    draw_curve=False,
                    draw_bbox=False,
                    domain_rand=False,
                    frame_rate=30,
                    frame_skip=1,
                    camera_width=640,
                    camera_height=480,
                    robot_speed=1.20, #MAXIMUM FORWARD ROBOT SPEED
                    accept_start_angle_deg=5,
                    full_transparency=False,
                    user_tile_start=None,
                    seed=None,
                    distortion=False,
                    randomize_maps_on_reset=False
)

obs = env.reset()
env.render()

standing_still = 0
done = False
while not done:
    lines, avg_lines, left_line, right_line = NLD.Perceive(obs)
    final = weighted_img(avg_lines, obs)

    steer, throttle = controler(left_line, right_line)
    if steer == 0 and throttle == 0:
        standing_still += 1
    else:
        standing_still = 0

    action = np.array([throttle, steer])
    obs, reward, done, info = env.step(action)
    env.render()
    cv2.imshow("Lines", lines)
    cv2.imshow("Averaged Line", cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)

    if standing_still == 100:
        done = True

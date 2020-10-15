#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
import os
os.sys.path.append("../gym-duckietown")

import cv2
import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
#from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.envs.duckietown_env import *
from gym_duckietown.wrappers import UndistortWrapper
from NaiveLaneDetection import NaiveLaneDetection

steer = 0
throttle = 0

# from experiments.utils import save_img
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

env.reset()
env.render()

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


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def controler(left_line, right_line):

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
        print("fudeu")
        steer = 0
        throttle = 0
    else:
        throttle = 0.4
        diff = d_left - d_right
        steer = -diff/5000

    return steer, throttle

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5
    
    obs, reward, done, info = env.step(action)
    lines, avg_lines, left_line, right_line = NLD.Perceive(obs)
    final = weighted_img(avg_lines, obs)

    steer, throttle = controler(left_line, right_line)
    action = np.array([throttle, steer])
    obs, reward, done, info = env.step(action)
    
    cv2.imshow("Lines", lines)
    cv2.imshow("Averaged Line", final)
    cv2.waitKey(1)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()

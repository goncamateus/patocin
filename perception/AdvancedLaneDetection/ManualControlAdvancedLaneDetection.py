#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
import os
os.sys.path.append("../../gym-duckietown")

import cv2
import sys
import argparse
import pyglet
import matplotlib.pyplot as plt
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
#from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.envs.duckietown_env import *
from gym_duckietown.wrappers import UndistortWrapper
from AdvancedLaneDetection import AdvancedLaneDetection

steer = 0
throttle = 0

# from experiments.utils import save_img
env = DuckietownLF(map_name='small_loop',
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

cv2.namedWindow("AdvancedLaneDetection")

points_src = [[0,300], [150, 140], [450, 140], [639, 300]] # left, apex_left, apex_right, right
points_dst = [[100, 479], [100,0], [400,0], [400,479]] # same order as src

yellow_threshold = [(20, 50, 100), (30, 255, 255)]
white_threshold = [(0, 0, 155), (255, 40, 255)]
ALD = AdvancedLaneDetection(points_src, points_dst, yellow_threshold, white_threshold)

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
print(key_handler)
env.unwrapped.window.push_handlers(key_handler)
it = 0

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global obs
    global it
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
    processed_image, hist = ALD.Perceive(cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))

    cv2.waitKey(1)
    cv2.imshow("AdvancedLaneDetection", processed_image)
    plt.plot(hist)
    #print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

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

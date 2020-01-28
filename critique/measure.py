import logging
from typing import Union
from statistics import mean
from math import degrees as deg
from collections import namedtuple

import cv2
import numpy as np

from critique import settings
from critique.utils import midpoint
from critique.pose.modules.pose import Pose, KEYPOINTS

class HEURISTICS():
    AVG_HIPS    = "AVG_HIPS"
    RIGHT_HIP   = "RIGHT_HIP"
    LEFT_HIP    = "LEFT_HIP"
    AVG_ANKLES  = "AVG_ANKLES"
    RIGHT_ANKLE = "RIGHT_ANKLE"
    LEFT_ANKLE  = "LEFT_ANKLE"
    AVG_ELBOWS  = "AVG_ELBOWS"
    RIGHT_ELBOW = "RIGHT_ELBOW"
    LEFT_ELBOW  = "LEFT_ELBOW"
    AVG_KNEES   = "AVG_KNEES"
    RIGHT_KNEE  = "RIGHT_KNEE"
    LEFT_KNEE   = "LEFT_KNEE"
    SIDE_NECK   = "SIDE_NECK"

def calc_floor_pt(pose:Pose):
    # TODO: depends if they are facing right or left
    return np.int_(midpoint(pose.keypoints[KEYPOINTS.L_ANK], pose.keypoints[KEYPOINTS.R_ANK]))

# TODO: Refactor with *args or **kwargs
def calc_angle(
        pose:Pose, 
        kpt1:Union[int, tuple, np.ndarray], 
        kpt2:Union[int, tuple, np.ndarray], 
        kpt3:Union[int, tuple, np.ndarray],
        degrees=False
    ):
    """
    Calculate angle between 3 keypoints using vectors.

    If the keypoint id arguments given are integers, they are taken directly from the pose,
    if they are a tuple, the average of the keypoint positions are taken,
    if they are a numpy array, use the value given.
    """

    if isinstance(kpt1, int):
        pt1 = pose.keypoints[kpt1]
        if pt1[0] == -1:
            return None
    elif isinstance(kpt1, tuple):
        if pose.keypoints[kpt1[0]][0] == -1 or pose.keypoints[kpt1[1]][0] == -1:
            return None
        pt1 = np.int_(midpoint(pose.keypoints[kpt1[0]], pose.keypoints[kpt1[1]]))
    elif isinstance(kpt1, np.ndarray):
        pt1 = kpt1
    else:
        raise TypeError(f"kpt1 must be of type int, tuple, or numpy array, not {type(kpt1)}")

    if isinstance(kpt2, int):
        pt2 = pose.keypoints[kpt2]
        if pt2[0] == -1:
            return None
    elif isinstance(kpt2, tuple):
        if pose.keypoints[kpt2[0]][0] == -1 or pose.keypoints[kpt2[1]][0] == -1:
            return None
        pt2 = np.int_(midpoint(pose.keypoints[kpt2[0]], pose.keypoints[kpt2[1]]))
    elif isinstance(kpt2, np.ndarray):
        pt2 = kpt2
    else:
        raise TypeError(f"kpt2 must be of type int, tuple, or numpy array, not {type(kpt2)}")

    if isinstance(kpt3, int):
        pt3 = pose.keypoints[kpt3]
        if pt3[0] == -1:
            return None
    elif isinstance(kpt3, tuple):
        if pose.keypoints[kpt3[0]][0] == -1 or pose.keypoints[kpt3[1]][0] == -1:
            return None
        pt3 = np.int_(midpoint(pose.keypoints[kpt3[0]], pose.keypoints[kpt3[1]]))
    else:
        raise TypeError(f"kpt3 must be of type int, tuple, or numpy array, not {type(kpt3)}")

    v1 = pt1 - pt2
    v2 = pt3 - pt2

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return None

    angle = np.arccos(v1.dot(v2) / (norm_v1 * norm_v2))

    if degrees:
        return deg(angle)
    return angle

def _right_hip(pose:Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.R_SHO, 
        KEYPOINTS.R_HIP, 
        KEYPOINTS.R_KNEE,
        degrees
    )

def _left_hip(pose:Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.L_SHO, 
        KEYPOINTS.L_HIP, 
        KEYPOINTS.L_KNEE,
        degrees
    )

def _right_ankle(pose:Pose, degrees=False):
    floor_pt = calc_floor_pt(pose)
    return calc_angle(
        pose,
        floor_pt, 
        KEYPOINTS.R_ANK, 
        KEYPOINTS.R_KNEE,
        degrees
    )

def _left_ankle(pose:Pose, degrees=False):
    floor_pt = calc_floor_pt(pose)
    return calc_angle(
        pose,
        floor_pt, 
        KEYPOINTS.L_ANK, 
        KEYPOINTS.L_KNEE,
        degrees
    )

def _right_elbow(pose:Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.R_SHO, 
        KEYPOINTS.R_ELB, 
        KEYPOINTS.R_WRI,
        degrees
    )

def _left_elbow(pose:Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.L_SHO, 
        KEYPOINTS.L_ELB, 
        KEYPOINTS.L_WRI,
        degrees
    )

def _right_knee(pose:Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.R_HIP, 
        KEYPOINTS.R_KNEE, 
        KEYPOINTS.R_ANK,
        degrees
    )

def _left_knee(pose:Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.L_HIP, 
        KEYPOINTS.L_KNEE, 
        KEYPOINTS.L_ANK,
        degrees
    )

def _side_neck(pose:Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.NOSE, 
        KEYPOINTS.NECK, 
        (KEYPOINTS.L_HIP, KEYPOINTS.R_HIP),
        degrees
    )

# Used to store a relative draw location
class DrawPosition:
    def __init__(self, kpt_ids, x_offset=0, y_offset=0): 
        self._kpt_ids = kpt_ids
        self._x_offset = x_offset
        self._y_offset = y_offset
        
    def pos(self, pose:Pose):
        x_pos = mean([
            pose.keypoints[kpt][0] for kpt in self._kpt_ids
        ])
        y_pos = mean([
            pose.keypoints[kpt][1] for kpt in self._kpt_ids
        ])
        return x_pos, y_pos


class PoseHeuristics():
    """Class to generate and manage pose heuristics."""

    # Heurisitics calculated using a helper function
    heuristic_funcs = {
        HEURISTICS.RIGHT_HIP: _right_hip,
        HEURISTICS.LEFT_HIP: _left_hip,
        HEURISTICS.RIGHT_ANKLE: _right_ankle,
        HEURISTICS.LEFT_ANKLE: _left_ankle,
        HEURISTICS.RIGHT_ELBOW: _right_elbow,
        HEURISTICS.LEFT_ELBOW: _left_elbow,
        HEURISTICS.RIGHT_KNEE: _right_knee,
        HEURISTICS.LEFT_KNEE: _left_knee,
        HEURISTICS.SIDE_NECK: _side_neck
    }

    # Heuristics that use an average of previously computed
    avg_heuristics = {
        HEURISTICS.AVG_ANKLES: (HEURISTICS.RIGHT_ANKLE, HEURISTICS.LEFT_ANKLE),
        HEURISTICS.AVG_ELBOWS: (HEURISTICS.RIGHT_ELBOW, HEURISTICS.LEFT_ELBOW),
        HEURISTICS.AVG_KNEES: (HEURISTICS.RIGHT_KNEE, HEURISTICS.LEFT_KNEE),
        HEURISTICS.AVG_HIPS: (HEURISTICS.RIGHT_HIP, HEURISTICS.LEFT_HIP)
    }

    # Used to compute where to draw the average labels
    draw_positions = {
        HEURISTICS.AVG_HIPS: DrawPosition([KEYPOINTS.R_HIP, KEYPOINTS.L_HIP]),
        HEURISTICS.RIGHT_HIP: DrawPosition([KEYPOINTS.R_HIP]),
        HEURISTICS.LEFT_HIP: DrawPosition([KEYPOINTS.L_HIP]),
        HEURISTICS.AVG_ANKLES: DrawPosition([KEYPOINTS.L_ANK, KEYPOINTS.R_ANK]),
        HEURISTICS.RIGHT_ANKLE: DrawPosition([KEYPOINTS.R_ANK]),
        HEURISTICS.LEFT_ANKLE: DrawPosition([KEYPOINTS.L_ANK]),
        HEURISTICS.AVG_ELBOWS: DrawPosition([KEYPOINTS.L_ELB, KEYPOINTS.R_ELB]),
        HEURISTICS.RIGHT_ELBOW: DrawPosition([KEYPOINTS.R_ELB]),
        HEURISTICS.LEFT_ELBOW: DrawPosition([KEYPOINTS.L_ELB]),
        HEURISTICS.AVG_KNEES: DrawPosition([KEYPOINTS.L_KNEE, KEYPOINTS.R_KNEE]),
        HEURISTICS.RIGHT_KNEE: DrawPosition([KEYPOINTS.R_KNEE]),
        HEURISTICS.LEFT_KNEE: DrawPosition([KEYPOINTS.L_KNEE]),
        HEURISTICS.SIDE_NECK: DrawPosition([KEYPOINTS.NECK]),
    }
    
    def __init__(self, pose:Pose=None, degrees=settings.DEGREES):
        self.pose = pose
        self.degrees = degrees

        self.heuristics = {}
        if pose is not None:
            self._generate_heuristics()
    
    def _generate_heuristics(self):
        for key, func in self.heuristic_funcs.items():
            val = func(self.pose, self.degrees)
            if val is not np.nan:
                self.heuristics[key] = val
            else:
                self.heuristics[key] = None 
        
        for key, kpts in self.avg_heuristics.items():
            h1 = self.heuristics[kpts[0]]
            h2 = self.heuristics[kpts[1]]

            if h1 is not None and h2 is not None:
                val = mean([
                    self.heuristics[kpts[0]],
                    self.heuristics[kpts[1]]
                ])
            else:
                val = None
            self.heuristics[key] = val

    def draw(self, img):
        for key, val in self.heuristics.items():
            if val is not None:
                dp = self.draw_positions[key]
                draw_pos = dp.pos(self.pose)
                cv2.putText(img, f"{key} {val:.2f}", draw_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
import logging
from typing import Union, Dict
import statistics
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
    AVG_SHLDRS  = "AVG_SHLDRS"
    RIGHT_SHLDR = "RIGHT_SHLDR"
    LEFT_SHLDR  = "LEFT_SHLDR"
    SIDE_NECK   = "SIDE_NECK"

class MV_DIRECTIONS:
    HOLD = 'HOLD'
    UP = 'UP'
    DOWN = 'DOWN'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'

def calc_floor_pt(pose: Pose):
    # TODO: depends if they are facing right or left
    return np.int_(midpoint(pose.keypoints[KEYPOINTS.L_ANK], pose.keypoints[KEYPOINTS.R_ANK]))

# TODO: Refactor with *args or **kwargs
def calc_angle(
        pose: Pose,
        kpt1:Union[int, tuple, np.ndarray],
        kpt2:Union[int, tuple, np.ndarray],
        kpt3:Union[int, tuple, np.ndarray],
        degrees=False,
        flip=False
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

    # angle = np.arccos(v1.dot(v2) / (norm_v1 * norm_v2))
    cross_product = np.cross(v1, v2)
    dot_product = np.dot(v1, v2)
    angle = np.arctan2(abs(cross_product), dot_product)

    if cross_product < 0:
        angle = 2*np.pi - angle
    
    if flip:
        angle = 2*np.pi - angle

    if degrees:
        return deg(angle)
    return angle

def _right_hip(pose: Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.R_SHO, 
        KEYPOINTS.R_HIP, 
        KEYPOINTS.R_KNEE,
        degrees,
        flip=True
    )

def _left_hip(pose: Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.L_SHO, 
        KEYPOINTS.L_HIP, 
        KEYPOINTS.L_KNEE,
        degrees
    )

def _right_ankle(pose: Pose, degrees=False):
    floor_pt = calc_floor_pt(pose)
    return calc_angle(
        pose,
        floor_pt, 
        KEYPOINTS.R_ANK, 
        KEYPOINTS.R_KNEE,
        degrees,
        flip=True
    )

def _left_ankle(pose: Pose, degrees=False):
    floor_pt = calc_floor_pt(pose)
    return calc_angle(
        pose,
        floor_pt, 
        KEYPOINTS.L_ANK, 
        KEYPOINTS.L_KNEE,
        degrees
    )

def _right_elbow(pose: Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.R_SHO, 
        KEYPOINTS.R_ELB, 
        KEYPOINTS.R_WRI,
        degrees,
        flip=True
    )

def _left_elbow(pose: Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.L_SHO, 
        KEYPOINTS.L_ELB, 
        KEYPOINTS.L_WRI,
        degrees
    )

def _right_knee(pose: Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.R_HIP, 
        KEYPOINTS.R_KNEE, 
        KEYPOINTS.R_ANK,
        degrees,
        flip=True
    )

def _left_knee(pose: Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.L_HIP, 
        KEYPOINTS.L_KNEE, 
        KEYPOINTS.L_ANK,
        degrees
    )

def _right_shldr(pose: Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.NECK,
        KEYPOINTS.R_SHO,
        KEYPOINTS.R_ELB, 
        degrees,
        flip=True
    )

def _left_shldr(pose: Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.NECK, 
        KEYPOINTS.L_SHO,
        KEYPOINTS.L_ELB,
        degrees
    )

def _side_neck(pose: Pose, degrees=False):
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
        
    def pos(self, pose: Pose):
        x_pos = mean([
            pose.keypoints[kpt][0] for kpt in self._kpt_ids
        ]) + self._x_offset
        y_pos = mean([
            pose.keypoints[kpt][1] for kpt in self._kpt_ids
        ]) + self._y_offset
        return x_pos, y_pos


class MovementVector:
    def __init__(self, kpt_id, hold_thresh=settings.MV_HOLD_THRESH, len_history=settings.MV_HISTORY):
        self._kpt_id = kpt_id
        self._x_history = []
        self._y_history = []
        self._hold_thresh = hold_thresh
        self._len_history = len_history

        self.x = MV_DIRECTIONS.HOLD
        self.y = MV_DIRECTIONS.HOLD
    
    def update(self, pose: Pose):
        kpt = pose.keypoints[self._kpt_id]

        if kpt[0] == -1:
            return False
        
        try:
            x_mean = mean(self._x_history)
            y_mean = mean(self._y_history)
        except statistics.StatisticsError:
            x_mean = kpt[0]
            y_mean = kpt[1]

        x_diff = x_mean - kpt[0]
        y_diff = y_mean - kpt[1]

        if x_diff > 0 and abs(x_diff) > self._hold_thresh:
            self.x = MV_DIRECTIONS.RIGHT
        elif x_diff < 0 and abs(x_diff) > self._hold_thresh:
            self.x = MV_DIRECTIONS.LEFT
        else:
            self.x = MV_DIRECTIONS.HOLD
        
        if y_diff > 0 and abs(y_diff) > self._hold_thresh:
            self.y = MV_DIRECTIONS.UP
        elif y_diff < 0 and abs(y_diff) > self._hold_thresh:
            self.y = MV_DIRECTIONS.DOWN
        else:
            self.y = MV_DIRECTIONS.HOLD

        self._x_history.append(kpt[0])
        self._y_history.append(kpt[1])

        if len(self._x_history) > self._len_history:
            self._x_history.pop(0)

        if len(self._y_history) > self._len_history:
            self._y_history.pop(0)

        return True


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
        HEURISTICS.RIGHT_SHLDR: _right_shldr,
        HEURISTICS.LEFT_SHLDR: _left_shldr,
        HEURISTICS.SIDE_NECK: _side_neck
    }

    # Heuristics that use an average of previously computed
    avg_heuristics = {
        HEURISTICS.AVG_ANKLES: (HEURISTICS.RIGHT_ANKLE, HEURISTICS.LEFT_ANKLE),
        HEURISTICS.AVG_ELBOWS: (HEURISTICS.RIGHT_ELBOW, HEURISTICS.LEFT_ELBOW),
        HEURISTICS.AVG_KNEES: (HEURISTICS.RIGHT_KNEE, HEURISTICS.LEFT_KNEE),
        HEURISTICS.AVG_HIPS: (HEURISTICS.RIGHT_HIP, HEURISTICS.LEFT_HIP),
        HEURISTICS.AVG_SHLDRS: (HEURISTICS.RIGHT_SHLDR, HEURISTICS.LEFT_SHLDR)
    }

    # Used to compute where to draw the average labels
    heuristics_draw_positions = {
        HEURISTICS.AVG_HIPS: DrawPosition([KEYPOINTS.R_HIP, KEYPOINTS.L_HIP]),
        HEURISTICS.RIGHT_HIP: DrawPosition([KEYPOINTS.R_HIP], y_offset=-20),
        HEURISTICS.LEFT_HIP: DrawPosition([KEYPOINTS.L_HIP], y_offset=-20),
        HEURISTICS.AVG_ANKLES: DrawPosition([KEYPOINTS.L_ANK, KEYPOINTS.R_ANK]),
        HEURISTICS.RIGHT_ANKLE: DrawPosition([KEYPOINTS.R_ANK]),
        HEURISTICS.LEFT_ANKLE: DrawPosition([KEYPOINTS.L_ANK]),
        HEURISTICS.AVG_ELBOWS: DrawPosition([KEYPOINTS.L_ELB, KEYPOINTS.R_ELB]),
        HEURISTICS.RIGHT_ELBOW: DrawPosition([KEYPOINTS.R_ELB], y_offset=-20),
        HEURISTICS.LEFT_ELBOW: DrawPosition([KEYPOINTS.L_ELB], y_offset=20),
        HEURISTICS.AVG_KNEES: DrawPosition([KEYPOINTS.L_KNEE, KEYPOINTS.R_KNEE]),
        HEURISTICS.RIGHT_KNEE: DrawPosition([KEYPOINTS.R_KNEE]),
        HEURISTICS.LEFT_KNEE: DrawPosition([KEYPOINTS.L_KNEE]),
        HEURISTICS.AVG_SHLDRS: DrawPosition([KEYPOINTS.L_SHO, KEYPOINTS.R_SHO], y_offset=-30),
        HEURISTICS.RIGHT_SHLDR: DrawPosition([KEYPOINTS.R_SHO], x_offset=-50, y_offset=-20),
        HEURISTICS.LEFT_SHLDR: DrawPosition([KEYPOINTS.L_SHO], x_offset=50, y_offset=20),
        HEURISTICS.SIDE_NECK: DrawPosition([KEYPOINTS.NECK]),
    }

    def __init__(self, pose:Pose=None, degrees=settings.DEGREES):
        self._curr_pose = pose
        self.degrees = degrees

        self.heuristics: Dict[str, float] = {} # TODO: refactor to angles
        self.movement_vectors: Dict[int, MovementVector] = {kpt: MovementVector(kpt) for kpt in KEYPOINTS.all()}

        self.update(pose)
    
    def _update_heuristics(self, pose:Pose=None):
        if pose is None:
            pose = self._curr_pose
        
        for key, func in self.heuristic_funcs.items():
            val = func(self._curr_pose, self.degrees)
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
    
    def _update_movement_vectors(self, pose:Pose=None):
        if pose is None:
            pose = self._curr_pose

        for key, mv in self.movement_vectors.items():
            mv.update(pose)

    def update(self, pose:Pose):
        self._curr_pose = pose
        if self._curr_pose is not None:
            self._update_heuristics()
            self._update_movement_vectors()

    def draw(self, img):
        for key, val in self.heuristics.items():
            if val is not None:
                dp = self.heuristics_draw_positions[key]
                draw_pos = dp.pos(self._curr_pose)
                cv2.putText(img, f"{key} {val:.2f}", draw_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
        
        for key, mv in self.movement_vectors.items():
            dp = self._curr_pose.keypoints[key]
            if dp[0] != -1:
                cv2.putText(img, f"{mv.x}", (dp[0], dp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255))
                cv2.putText(img, f"{mv.y}", (dp[0], dp[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255))

    def get_angle(self, heuristic_id: int):
        return self.heuristics[heuristic_id]
    
    def get_movement(self, kpt_id: int):
        return self.movement_vectors[kpt_id]
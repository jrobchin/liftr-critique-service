import logging
from typing import Union
from statistics import mean
from math import degrees as deg

import cv2
import numpy as np

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
    NECK        = "NECK"

def calc_floor_pt(pose:Pose):
    # TODO: depends if they are facing right or left
    return np.int_(midpoint(pose.keypoints[KEYPOINTS.L_ANK], pose.keypoints[KEYPOINTS.R_ANK]))

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
    elif isinstance(kpt1, tuple):
        pt1 = np.int_(midpoint(pose.keypoints[kpt1[0]], pose.keypoints[kpt1[1]]))
    elif isinstance(kpt1, np.ndarray):
        pt1 = kpt1
    else:
        raise TypeError("kpt1 must be of type int, tuple, or numpy array")

    if isinstance(kpt2, int):
        pt2 = pose.keypoints[kpt2]
    elif isinstance(kpt2, tuple):
        pt2 = np.int_(midpoint(pose.keypoints[kpt2[0]], pose.keypoints[kpt2[1]]))
    elif isinstance(kpt2, np.ndarray):
        pt2 = kpt2
    else:
        raise TypeError("kpt2 must be of type int, tuple, or numpy array")

    if isinstance(kpt3, int):
        pt3 = pose.keypoints[kpt3]
    elif isinstance(kpt3, tuple):
        pt3 = np.int_(midpoint(pose.keypoints[kpt3[0]], pose.keypoints[kpt3[1]]))
    elif isinstance(kpt3, np.ndarray):
        pt3 = kpt3
    else:
        raise TypeError("kpt3 must be of type int, tuple, or numpy array")

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

def _avg_hips(pose:Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.NECK, 
        (KEYPOINTS.R_HIP, KEYPOINTS.L_HIP), 
        (KEYPOINTS.R_KNEE, KEYPOINTS.L_KNEE),
        degrees
    )

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

def _avg_ankles(pose:Pose, degrees=False):
    floor_pt = calc_floor_pt(pose)
    return calc_angle(
        pose,
        floor_pt, 
        (KEYPOINTS.R_ANK, KEYPOINTS.L_ANK), 
        (KEYPOINTS.R_KNEE, KEYPOINTS.L_KNEE),
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

def _avg_elbows(pose:Pose, degrees=False):
    return calc_angle(
        pose,
        (KEYPOINTS.R_SHO, KEYPOINTS.L_SHO), 
        (KEYPOINTS.R_ELB, KEYPOINTS.L_ELB), 
        (KEYPOINTS.R_WRI, KEYPOINTS.L_WRI),
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

def _avg_knees(pose:Pose, degrees=False):
    return calc_angle(
        pose,
        (KEYPOINTS.R_HIP, KEYPOINTS.L_HIP), 
        (KEYPOINTS.R_KNEE, KEYPOINTS.L_KNEE), 
        (KEYPOINTS.R_ANK, KEYPOINTS.L_ANK),
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

def _neck(pose:Pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.NOSE, 
        KEYPOINTS.NECK, 
        (KEYPOINTS.L_HIP, KEYPOINTS.R_HIP),
        degrees
    )


class PoseHeuristics():
    """Class to generate and manage pose heuristics."""

    heuristic_funcs = {
        HEURISTICS.AVG_HIPS: _avg_hips,
        HEURISTICS.RIGHT_HIP: _right_hip,
        HEURISTICS.LEFT_HIP: _left_hip,
        HEURISTICS.AVG_ANKLES: _avg_ankles,
        HEURISTICS.RIGHT_ANKLE: _right_ankle,
        HEURISTICS.LEFT_ANKLE: _left_ankle,
        HEURISTICS.AVG_ELBOWS: _avg_elbows,
        HEURISTICS.RIGHT_ELBOW: _right_elbow,
        HEURISTICS.LEFT_ELBOW: _left_elbow,
        HEURISTICS.AVG_KNEES: _avg_knees,
        HEURISTICS.RIGHT_KNEE: _right_knee,
        HEURISTICS.LEFT_KNEE: _left_knee,
        HEURISTICS.NECK: _neck
    }

    draw_locations = {
        HEURISTICS.AVG_HIPS: (KEYPOINTS.R_HIP, KEYPOINTS.L_HIP),
        HEURISTICS.RIGHT_HIP: KEYPOINTS.R_HIP,
        HEURISTICS.LEFT_HIP: KEYPOINTS.L_HIP,
        HEURISTICS.AVG_ANKLES: (KEYPOINTS.L_ANK, KEYPOINTS.R_ANK),
        HEURISTICS.RIGHT_ANKLE: KEYPOINTS.R_ANK,
        HEURISTICS.LEFT_ANKLE: KEYPOINTS.L_ANK,
        HEURISTICS.AVG_ELBOWS: (KEYPOINTS.L_ELB, KEYPOINTS.R_ELB),
        HEURISTICS.RIGHT_ELBOW: KEYPOINTS.R_ELB,
        HEURISTICS.LEFT_ELBOW: KEYPOINTS.L_ELB,
        HEURISTICS.AVG_KNEES: (KEYPOINTS.L_KNEE, KEYPOINTS.R_KNEE),
        HEURISTICS.RIGHT_KNEE: KEYPOINTS.R_KNEE,
        HEURISTICS.LEFT_KNEE: KEYPOINTS.L_KNEE,
        HEURISTICS.NECK: KEYPOINTS.NECK
    }
    
    def __init__(self, pose:Pose=None, degrees=False):
        self.pose = pose
        self.degrees = degrees

        self.heuristics = {}
        if pose is not None:
            self._generate_heuristics()
    
    def _generate_heuristics(self):
        for key, func in self.heuristic_funcs.items():
            logging.debug((key, func))
            val = func(self.pose, self.degrees)
            if val is not np.nan:
                self.heuristics[key] = val
            else:
                self.heuristics[key] = None 
    
    def draw(self, img):
        for key, val in self.heuristics.items():
            if val is not None:
                draw_keypoint = self.draw_locations[key]
                if isinstance(draw_keypoint, tuple):
                    draw_pos = np.int_(midpoint(
                                self.pose.keypoints[draw_keypoint[0]], 
                                self.pose.keypoints[draw_keypoint[1]]
                               ))
                else:
                    draw_pos = self.pose.keypoints[draw_keypoint]
                draw_pos = tuple(draw_pos.tolist())
                cv2.putText(img, f"{key} {val:.2f}", draw_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
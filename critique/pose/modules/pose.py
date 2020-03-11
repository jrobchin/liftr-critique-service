import copy

import cv2
import numpy as np

from critique.pose.modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS

class KEYPOINTS():
    NOSE   = 0
    NECK   = 1
    R_SHO  = 2
    R_ELB  = 3
    R_WRI  = 4
    L_SHO  = 5
    L_ELB  = 6
    L_WRI  = 7
    R_HIP  = 8
    R_KNEE = 9
    R_ANK  = 10
    L_HIP  = 11
    L_KNEE = 12
    L_ANK  = 13
    R_EYE  = 14
    L_EYE  = 15
    R_EAR  = 16
    L_EAR  = 17
    
    NUM_KPTS = 18

    @classmethod
    def all(cls):
        return [i for i in range(cls.NUM_KPTS)]


class Pose():
    num_kpts = 18
    kpts = KEYPOINTS
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    kpt_groups = {
        'left': [
            KEYPOINTS.NOSE,
            KEYPOINTS.NECK,
            KEYPOINTS.L_SHO,
            KEYPOINTS.L_ELB,
            KEYPOINTS.L_WRI,
            KEYPOINTS.L_HIP,
            KEYPOINTS.L_KNEE,
            KEYPOINTS.L_ANK,
            KEYPOINTS.L_EYE,
            KEYPOINTS.L_EAR
        ],
        'right': [
            KEYPOINTS.NOSE,
            KEYPOINTS.NECK,
            KEYPOINTS.R_SHO,
            KEYPOINTS.R_ELB,
            KEYPOINTS.R_WRI,
            KEYPOINTS.R_HIP,
            KEYPOINTS.R_KNEE,
            KEYPOINTS.R_ANK,
            KEYPOINTS.R_EYE,
            KEYPOINTS.R_EAR
        ]
    }
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1

    def __init__(self, keypoints, confidence, color=[0, 224, 255]):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(keypoints.shape[0]):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        self.bbox = cv2.boundingRect(found_keypoints)
        self.id = None
        self.color = color

    
    def __eq__(self, other):
        return np.array_equal(self.keypoints, other.keypoints)

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img, kpt_id_labels=False, kpt_coords=False):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, self.color, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, self.color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), self.color, 2)

        if kpt_id_labels:
            for kpt_id in range(self.num_kpts):
                kpt_pos = self.keypoints[kpt_id]
                cv2.putText(img, f"{kpt_id} {self.kpt_names[kpt_id]}", (int(kpt_pos[0]), int(kpt_pos[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        if kpt_coords:
            for kpt_id in range(self.num_kpts):
                if "l_" not in self.kpt_names[kpt_id]:
                    continue
                kpt_pos = self.keypoints[kpt_id]
                cv2.putText(img, f"{self.kpt_names[kpt_id]} {int(kpt_pos[0]), int(kpt_pos[1])}", (int(kpt_pos[0]), int(kpt_pos[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


    def get_keypoint_index(self, name):
        try:
            return self.kpt_names.index(name)
        except ValueError:
            raise ValueError(f"`{name}` not in keypoint names")

    def get_keypoint(self, name):
        index = self.get_keypoint_index(name)
        return self.keypoints[index]
    
    def to_dict(self):
        d = {}
        for name, val in zip(self.kpt_names, self.keypoints):
            d[name] = val
        return d

    def get_kpt_group(self, group=None):
        """
        Return new pose with only keypoints in `group`.
        Must be one of `left` or `right`.
        """
        if group is not None:
            keypoints = copy.copy(self.keypoints) # TODO: maybe inefficient to copy and then iterate
            for i, kpt in enumerate(keypoints):
                if i not in self.kpt_groups[group]:
                    keypoints[i] = [-1, -1]
            return Pose(keypoints, self.confidence, self.color)
        else:
            return self


def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def propagate_ids(previous_poses, current_poses, threshold=3):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :return: None
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose_id in range(len(current_poses)):
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for previous_pose_id in range(len(previous_poses)):
            if not mask[previous_pose_id]:
                continue
            iou = get_similarity(current_poses[current_pose_id], previous_poses[previous_pose_id])
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_poses[previous_pose_id].id
                best_matched_id = previous_pose_id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_poses[current_pose_id].update_id(best_matched_pose_id)

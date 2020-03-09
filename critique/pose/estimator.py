import os
from typing import List

import cv2
import numpy as np
import torch

from critique import settings
from critique.utils import COLORS
from critique.settings import CHECKPOINT_PATH
from critique.pose.models.with_mobilenet import PoseEstimationWithMobileNet
from critique.pose.modules import keypoints, pose as pose_module
from critique.pose import preprocessing
from critique.pose.modules.pose import Pose

HEIGHT_SIZE = 256
STRIDE = 8
UPSAMPLE_RATIO = 4

class PoseEstimator:

    def __init__(self, height_size=256):
        self.height_size = height_size

        if not settings.DISABLE_NET:
            self._init_net()

    def _init_net(self):
        self.net = PoseEstimationWithMobileNet()
        self.net.eval()
        self.net = self.net.cuda()

        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        self.net.load_checkpoint(checkpoint)
    
    def _infer(self, img, net_input_height_size=HEIGHT_SIZE, stride=STRIDE, upsample_ratio=UPSAMPLE_RATIO,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
        height, width, _ = img.shape
        scale = net_input_height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_img = preprocessing.normalize(scaled_img, img_mean, img_scale)
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = preprocessing.pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        tensor_img = tensor_img.cuda()

        stages_output = self.net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad
    
    def estimate(self, img, conf_thresh=0) -> List[Pose]:
        if settings.DISABLE_NET:
            return []

        heatmaps, pafs, scale, pad = self._infer(img)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(pose_module.Pose.num_kpts):
            total_keypoints_num += keypoints.extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = keypoints.group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]): # pylint: disable=E1136  # pylint/issues/3139
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * STRIDE / UPSAMPLE_RATIO - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * STRIDE / UPSAMPLE_RATIO - pad[0]) / scale
        current_poses = []

        color_num = 0
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((pose_module.Pose.num_kpts, 2), dtype=np.int32) * -1
            for kpt_id in range(pose_module.Pose.num_kpts):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = pose_module.Pose(pose_keypoints, pose_entries[n][18], COLORS[color_num])
            color_num = (color_num + 1) % len(COLORS)
            if pose.confidence > conf_thresh:
                current_poses.append(pose)

        return current_poses
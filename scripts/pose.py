import os
import pickle
import logging

import cv2
import click
import ffmpeg
import numpy as np

from critique.pose.modules.pose import KEYPOINTS
from critique.io import ImageReader, VideoReader
from critique.pose.estimator import PoseEstimator
from critique.measure import PoseHeuristics, calc_floor_pt

@click.group()
@click.option('--log-level', help="Set the log level.")
def cli(log_level):
    if log_level:
        logging.basicConfig(level=log_level.upper())

@cli.command()
@click.argument('image')
@click.argument('output')
def get_pose(image, output):
    frame_provider = ImageReader(image)

    estimator = PoseEstimator()

    for frame in frame_provider:
        poses = estimator.estimate(frame)
    
    output_ext = output.split('.')[-1]
    if output_ext == 'json':
        pass
    elif output_ext == 'pkl':
        with open(output, 'wb') as f:
            pickle.dump(poses, f)
    else:
        raise ValueError(f"output file extension `{output_ext}` is not supported")
        
@cli.command()
@click.argument('image')
@click.argument('output')
def draw_pose(image, output):
    frame_provider = ImageReader(image)

    estimator = PoseEstimator()

    for frame in frame_provider:
        poses = estimator.estimate(frame)
        for pose in poses:
            pose.draw(frame)
        cv2.imwrite(output, frame)

@cli.command()
@click.argument('video')
@click.argument('output')
def draw_pose_video(video, output):
    frame_provider = VideoReader(video)

    estimator = PoseEstimator()
    logging.debug("Instantiated pose estimator")
    render_process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f"{frame_provider.width}x{frame_provider.height}")
        .output(output, pix_fmt='rgb24')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    logging.debug("Started render process")

    for i, frame in enumerate(frame_provider):
        logging.debug(f"Start frame {i}...")
        poses = estimator.estimate(frame)
        for pose in poses:
            pose.draw(frame, kpt_id_labels=True)
        render_process.stdin.write(
            frame
            .astype(np.uint8)
            .tobytes()
        )
        logging.debug(f"Finish frame {i}...")

@cli.command()
@click.argument('image')
def show_floor_pt(image):
    img = cv2.imread(image)
    estimator = PoseEstimator()
    poses = estimator.estimate(img, 5)

    for pose in poses:
        floor_pt = calc_floor_pt(pose)
        print(pose.confidence)
        print(floor_pt)
        print(pose.keypoints[KEYPOINTS.L_ANK])
        print(pose.keypoints[KEYPOINTS.R_ANK])
        cv2.circle(img, tuple(floor_pt.tolist()), 3, (255, 0, 255), -1)
        cv2.circle(img, tuple(pose.keypoints[KEYPOINTS.L_ANK].tolist()), 3, (255, 0, 255), -1)
        cv2.circle(img, tuple(pose.keypoints[KEYPOINTS.R_ANK].tolist()), 3, (255, 0, 255), -1)
    
    cv2.imshow('frame', img)
    cv2.waitKey(0)


@cli.command()
@click.argument('image')
@click.option('-o', '--output', help='path to save pose heuristics to')
@click.option('-d', '--draw', is_flag=True, help='path to draw heuristics to')
@click.option('--degrees', is_flag=True, help='flag to format in degrees')
def get_heuristics(image, output, draw, degrees):
    frame_provider = ImageReader(image)
    estimator = PoseEstimator()

    frame = frame_provider.get_image()

    heuristics = []

    poses = estimator.estimate(frame)
    for pose in poses:
        print(pose.confidence)
        print(pose.keypoints)
        pose.draw(frame)
        ph = PoseHeuristics(pose, degrees)
        heuristics.append(ph)
    
    if not output:
        for h in heuristics:
            print(h.heuristics)
    
    if draw:
        for h in heuristics:
            h.draw(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)


@cli.command()
@click.argument('video')
@click.option('--degrees', is_flag=True, help='flag to format in degrees')
@click.option('--kpt-filter', type=click.Choice(['left', 'right'], case_sensitive=False), help='filter to apply to keypoints')
def get_heuristics_video(video, degrees, kpt_filter):
    frame_provider = VideoReader(video)
    estimator = PoseEstimator()

    for frame in frame_provider:
        heuristics = []

        poses = estimator.estimate(frame)
        for pose in poses:
            pose = pose.get_kpt_group(kpt_filter)
            pose.draw(frame)
            ph = PoseHeuristics(pose, degrees)
            heuristics.append(ph)
        
        for h in heuristics:
            h.draw(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


@cli.command()
@click.argument('camera')
@click.option('--degrees', is_flag=True, help='flag to format in degrees')
@click.option('--kpt-filter', type=click.Choice(['left', 'right'], case_sensitive=False), help='filter to apply to keypoints')
def live_pose(camera, degrees, kpt_filter):
    frame_provider = VideoReader(int(camera))
    estimator = PoseEstimator()

    for frame in frame_provider:
        heuristics = []

        poses = estimator.estimate(frame)
        for pose in poses:
            pose = pose.get_kpt_group(kpt_filter)
            pose.draw(frame)
            ph = PoseHeuristics(pose, degrees)
            heuristics.append(ph)
        
        for h in heuristics:
            h.draw(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    cli() # pylint: disable=no-value-for-parameter
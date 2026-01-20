#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cv2
bf = cv2.BFMatcher(cv2.NORM_L2)
from tqdm import tqdm
import os
import math
import time
import platform
import matplotlib
matplotlib.use("Agg")  # must be before pyplot import
import matplotlib.pyplot as plt

from pyslam.config import Config

from pyslam.slam.visual_odometry import VisualOdometryEducational
from pyslam.slam.visual_odometry_rgbd import (
    VisualOdometryRgbd,
    VisualOdometryRgbdTensor,
)
from pyslam.slam.camera import PinholeCamera
from pyslam.io.silk_masker import SilkMaskGenerator, MaskLoader
from pyslam.io.ground_truth import groundtruth_factory
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import DatasetType, SensorType
import wandb
from pyslam.viz.mplot_thread import Mplot2d, Mplot3d
# from pyslam.viz.qplot_thread import Qplot2d
from pyslam.viz.rerun_interface import Rerun

from pyslam.local_features.feature_tracker import (
    feature_tracker_factory,
    FeatureTrackerTypes,
)
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.utilities.utils_sys import Printer
import torch

import silk.icra25.frame_score as fscore
from silk.icra25.featureness import load_images

import pickle
import sys
kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kResultsFolder = kRootFolder + "/results"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--viz",
    action="store_true",
    help="enable visualization",
)
args = parser.parse_args()

kVisualize = args.viz

kUseRerun = True
# check rerun does not have issues
# if kUseRerun and not Rerun.is_ok:
#     kUseRerun = False

"""
use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
"""

import subprocess
from pathlib import Path
import re

def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Accepts HxW, HxWx1, HxWx3 in float/uint8; returns HxWx3 uint8."""
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    if img.dtype != np.uint8:
        # assume [0,1] or arbitrary float; robust normalize to [0,255]
        x = img.astype(np.float32)
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        x = (x * 255.0).clip(0, 255)
        img = x.astype(np.uint8)

    return img

def _mask_to_uint8_rgb(mask: np.ndarray) -> np.ndarray:
    """Accepts HxW or HxWx1 mask in {0,1} or [0,1] or uint8; returns white mask on black."""
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.dtype != np.uint8:
        m = mask.astype(np.float32)
        # treat anything >0.5 as foreground if looks like probs
        if m.max() <= 1.0:
            m = (m > 0.5).astype(np.float32)
        m = (m * 255.0).clip(0, 255).astype(np.uint8)
    else:
        # if it's 0/1, scale up
        if mask.max() <= 1:
            m = (mask * 255).astype(np.uint8)
        else:
            m = mask
    m3 = np.stack([m, m, m], axis=-1)
    return m3

def make_1x3_grid(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Returns a single Hx(3W)x3 uint8 image: [image | mask | overlay].
    overlay = image*(1-alpha) + mask_color*alpha (mask shown in green).
    """
    img = _to_uint8_rgb(image)
    m3 = _mask_to_uint8_rgb(mask)

    # green mask color (only where mask is on)
    green = np.zeros_like(img)
    green[..., 1] = m3[..., 0]  # use mask intensity as green channel

    overlay = (img.astype(np.float32) * (1 - alpha) + green.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

    # concat horizontally
    grid = np.concatenate([img, m3, overlay], axis=1)
    return grid


def extract_xz(poses):
    """
    poses: list or np.ndarray of shape (N, 7)
    returns: (N, 2) array [x, z]
    """
    poses = np.asarray(poses)
    # x = poses[:, 0]
    # z = poses[:, 2]
    x = poses[:, 3]
    z = poses[:, 11]
    return x, z

def make_xz_traj_figure(evo_cands, evo_gts, title=None):
    x_est, z_est = extract_xz(evo_cands)
    x_gt,  z_gt  = extract_xz(evo_gts)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(x_gt, z_gt, label="GT", linewidth=2)
    ax.plot(x_est, z_est, label="Est", linewidth=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()

    if title is not None:
        ax.set_title(title)

    return fig


def factory_plot2d(*args, **kwargs):
    if kVisualize:
        if kUseRerun:
            return None
        else:
            return Mplot2d(*args, **kwargs)
    else:
        return None

def evo_ape_rmse_kitti(est_path, gt_path, monocular=False) -> float:
    """
    Runs evo_ape kitti and returns the 'rmse' value from its stdout.
    """
    align = ["-a"]
    if monocular:
        align += ["--correct_scale"]

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    # capture stdout so we can parse rmse
    out = subprocess.check_output(
        ["evo_ape", "kitti", str(gt_path), str(est_path), *align],
        env=env,
        text=True,
        stderr=subprocess.STDOUT,
    )

    # evo prints a table that includes a line like: "rmse 0.1234"
    m = re.search(r"^\s*rmse\s+([0-9]*\.?[0-9]+([eE][-+]?\d+)?)\s*$", out, re.MULTILINE)
    if not m:
        raise RuntimeError(f"Could not parse rmse from evo_ape output.\n--- evo_ape output ---\n{out}")
    return float(m.group(1))
    

def process_data(results: dict, 
                 images=None,
                 masks=None, 
                 save_evo_report=True,
                 save_pkl=True,
                 draw_tracks=False, 
                 plot_traj=True, 
                 traj_skips=1):

    

    exp_name = results['exp_name']
    name = results['feature_name']
    matched_kps = results['matched_kps']
    num_inliers = results['num_inliers']
    px_shifts = results['px_shifts']
    rs = results['rs']
    ts = results['ts']
    kps = results['kps']
    des = results['des']        
    original_kps = results['original_kps']
    masked_kps = results['masked_kps']
    xs = results['xs']
    ys = results['ys']
    zs = results['zs']
    gtxs = results['gtxs']
    gtys = results['gtys']
    gtzs = results['gtzs']
    est_times = results['est_times']
    loop_total_time = results['total_loop_time']
    prob_thresh = results['prob_thresh']
    uncer_thresh = results['uncer_thresh']

    evo_cands = results['evo_cands']
    evo_gts = results['evo_gts']

    run_folder = os.path.join(kResultsFolder, exp_name)
    os.makedirs(run_folder, exist_ok=True)

    if save_pkl:
        with open(f'{run_folder}/{exp_name}.pkl', 'wb') as f:
            pickle.dump(results, f)


    print_str = f'''
        matched_kps: {np.mean(matched_kps)}
        num_inliers: {np.mean(num_inliers)}
        px_shifts: {np.mean(px_shifts)}
        rs: {len(rs)}
        ts: {len(ts)}
        xs: {len(xs)}
        ys: {len(ys)}
        zs: {len(zs)}
        gtxs: {len(gtxs)}
        gtys: {len(gtys)}
        gtzs: {len(gtzs)}
        kps: {len(kps)}
        des: {len(des)}
        images: {len(images)}
        masks: {len(masks)},
        original_kps: {np.sum([len(kp) for kp in original_kps])/len(original_kps)}
        masked_kps: {np.sum([len(kp) for kp in masked_kps])/len(masked_kps)}
        est_times: {np.sum(est_times)/len(est_times)}
        evo_cands: {len(evo_cands)}
        evo_gts: {len(evo_gts)}
        total_loop_time: {loop_total_time} seconds
        prob_thresh: {prob_thresh}
        uncer_thresh: {uncer_thresh}
            '''
    print(print_str)
    with open(f'{run_folder}/{exp_name}_stats.txt', 'w') as f:
        f.write(print_str)

    if draw_tracks:
        idxs = range(len(matched_kps))
        # idxs = [i*dataset_skip for i in idxs]
        fig, ax = plt.subplots(2,1)
        ax[0].plot(idxs, matched_kps, label='matched_kps')
        ax[0].plot(idxs, num_inliers, label='num_inliers')
        ax[0].legend()
        ax[1].plot(idxs, px_shifts, label='px_shifts')
        ax[1].legend()
        plt.savefig(f'{run_folder}/{exp_name}_kp_inliers.png')
        plt.close()

        # Initialize an empty list to store matches between consecutive frames
        matches_list = []

        # Loop over consecutive pairs of frames
        for i in range(len(des) - 1):
            des1 = des[i]
            des2 = des[i + 1]

            # If either descriptor is None, we simply append an empty match list
            if des1 is None or des2 is None:
                matches = []
            else:
                # Match features using BFMatcher and sort the matches based on distance
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
            
            # Append the matches corresponding to frame i (matched with frame i+1)
            matches_list.append(matches)

        tracks = {}
        next_track_id = 0

        for i in range(len(des)):
            if des[i] is not None and des[i].dtype != np.float32:
                des[i] = des[i].astype(np.float32)

        # Process each frame as a starting point
        for start_frame in tqdm(range(len(images) - 1)):
            # Initialize new tracks for features in this frame
            current_features = {i: next_track_id + i for i in range(len(kps[start_frame]))}
            
            # Add new features to tracks
            for i in range(len(kps[start_frame])):
                tracks[next_track_id + i] = [(start_frame, i)]
            
            next_track_id += len(kps[start_frame])

            # Track these features in subsequent frames
            prev_descriptors = des[start_frame]
            prev_features = current_features

            if prev_descriptors is None:
                continue  # Skip frames without descriptors

            for frame_idx in range(start_frame + 1, len(images)):
                current_descriptors = des[frame_idx]

                if current_descriptors is None:
                    continue  # Skip frames without descriptors

                # Ensure descriptors are of the same type
                if prev_descriptors.dtype != np.float32:
                    prev_descriptors = prev_descriptors.astype(np.float32)
                if current_descriptors.dtype != np.float32:
                    current_descriptors = current_descriptors.astype(np.float32)

                matches = bf.knnMatch(prev_descriptors, current_descriptors, k=2)

                # Apply Lowe’s ratio test
                good_matches = []
                for match in matches:
                    if len(match) >= 2:
                        m, n = match[:2]
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                # Update tracks with good matches
                new_features = {}
                for match in good_matches:
                    query_idx = match.queryIdx  # Previous frame feature
                    train_idx = match.trainIdx  # Current frame feature
                    
                    if query_idx in prev_features:
                        track_id = prev_features[query_idx]
                        tracks[track_id].append((frame_idx, train_idx))
                        new_features[train_idx] = track_id  # Carry forward to next frame

                # Prepare for the next frame
                prev_descriptors = current_descriptors
                prev_features = new_features

        plt.figure(figsize=(24, 12))
        for track_id, track in tqdm(tracks.items()):
            frames = [f for f, _ in track]
            plt.plot([track_id] * len(frames), frames, marker='o', linestyle='-', lw=0.05)


        plt.xlabel('Feature Track ID ')
        plt.ylabel('Frame ID')
        plt.title(f'Feature Tracks Over Multiple Frames - {name}')
        plt.gca().invert_yaxis() 
        plt.savefig(f'{run_folder}/{exp_name}_tracks.png')
        plt.close()

        with open(f'{run_folder}/{exp_name}_tracks.pkl', 'wb') as f:
            pickle.dump(tracks, f)

        

    xs_p = xs[::traj_skips]
    zs_p = zs[::traj_skips]
    gtxs_p = gtxs[::traj_skips]
    gtzs_p = gtzs[::traj_skips]
    
    if plot_traj:
        plt.figure(figsize=(12, 12))
        plt.plot(xs_p, zs_p, c='tab:red', label='estimated')
        plt.plot(gtxs_p, gtzs_p, c='tab:green', label='ground truth')
        plt.scatter(xs_p, zs_p, c='tab:red', s=2)
        plt.scatter(gtxs_p, gtzs_p, c='tab:green', s=2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(f'2D Trajectory - {name}')
        plt.legend()
        plt.savefig(f"{run_folder}/{exp_name}_2d.png")
        plt.close()

    print(f'@@@@@@@@@@@@@@@@ SAVED TRAJ AT {run_folder}/{exp_name}_trajectory.csv')

    with open(f'{run_folder}/{exp_name}_trajectory.csv', 'w') as f:
        print(f"saving {run_folder}/{exp_name}_2d.csv with {len(xs)} points")
        f.write(f'i,x,y,z,gtx,gty,gtz\n')
        for i in range(len(xs)):
            # f.write(f'{xs[i]},{ys[i]},{zs[i]},{gtxs[i]},{gtys[i]},{gtzs[i]}\n')
            f.write(f'{i},{xs[i]},{ys[i]},{zs[i]},{gtxs[i]},{gtys[i]},{gtzs[i]}\n')

    if save_evo_report:
        np.savetxt(f'{run_folder}/{exp_name}_evo_cands.txt', np.array(evo_cands), fmt='%.9f')
        np.savetxt(f'{run_folder}/{exp_name}_evo_gts.txt', np.array(evo_gts), fmt='%.9f')

        evo_report_dir = generate_evo_report(
            evo_cand=f'{run_folder}/{exp_name}_evo_cands.txt',
            evo_gt=f'{run_folder}/{exp_name}_evo_gts.txt',
            out_dir=f'{run_folder}/{exp_name}_evo_report',
            monocular=True,
        )
        print(f'@@@@@@@@@@@@@@@@ SAVED EVO REPORT AT {evo_report_dir}')

    return
    


def run_exp(
        exp_name: str = "",
        feature_type = FeatureTrackerConfigs.LK_SHI_TOMASI,
        feature_name: str = "LK_SHI_TOMASI",
        is_baseline: bool = False,
        feature_num: int = 2000,
        max_images: int = -1,
        save_intermediate: bool = False,
        save_pkl: bool = False,
        save_evo_report: bool = True,
        plot_tracks: bool = False,
        plot_traj: bool = False,
        prob_thresh: float = 0.0,
        uncer_thresh: float = 0.1,
        optuna_mode: bool = False,
        base_rmse: float = None,
        ):
    
    if os.environ.get('PYSLAM_CONFIG') is None:
        raise RuntimeError("Please set the PYSLAM_CONFIG with the same scope as this script!")
    
    config_loc = os.environ.get('PYSLAM_CONFIG')
    config_name = config_loc.split('.')[0]

    if exp_name == "":
        exp_name = f'{config_name}_#{feature_name}#_@{feature_num}@' 
    else:
        # exp_name = f'{exp_name}_#{feature_name}#_@{feature_num}@'
        pass

    wandb_run = None
    try:
       
    
        config = Config()

        dataset = dataset_factory(config)

        mask_gen = None

        if not os.path.exists(f'{kResultsFolder}/logs'):
            os.makedirs(f'{kResultsFolder}/logs')

        

        wandb_run = None

        if not is_baseline:
            mask_gen = SilkMaskGenerator(
                dnn_ckpt=f"{kScriptFolder}/silk_data/dnn.ckpt",
                uh_ckpt=f"{kScriptFolder}/silk_data/uh_mc100.ckpt",
                prob_thresh=prob_thresh,
                uncer_thresh=uncer_thresh,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # dummy_image = f"{kScriptFolder}/silk_data/frame.png"
            # img_t = cv2.imread(dummy_image)
            # if img_t is None:
            #     raise RuntimeError(f"Could not read dummy image at {dummy_image}")

            # mask = mask_gen(img_t)

            img_t = dataset.getImage(30)
            mask = mask_gen(img_t)

            # SILK LOAD PRECOMPUTED
            # mask_gen = MaskLoader(
            #     mean_location=f"{kScriptFolder}/silk_data/mean_maps/",
            #     var_location=f"{kScriptFolder}/silk_data/var_maps/",
            #     prob_val=prob_thresh,
            #     unc_val=uncer_thresh,
            #     dummy_image=img_t,
            #     device="cuda" if torch.cuda.is_available() else "cpu"
            # )
            # mask = mask_gen("/home/christoa/Downloads/torrents/data_odometry_gray/dataset/sequences/00/mean/000000_mean.npy","/home/christoa/Downloads/torrents/data_odometry_gray/dataset/sequences/00/var/000000_var.npy")


            image_area = img_t.shape[0] * img_t.shape[1]
            non_zero_area = int(np.sum(mask > 0))
            frac = non_zero_area / float(image_area)

            if frac < 0.01:
                dummy_ok = False
                dummy_reason = f"too_few_features frac={frac:.6f}"
                Printer.yellow(
                    f"[Dummy FAIL] too few features ({non_zero_area}/{image_area}={frac*100:.4f}%) "
                    f"p={prob_thresh}, u={uncer_thresh}"
                )
                raise ValueError("Dummy mask test failed: too few features.")
            elif frac > 0.90:
                dummy_ok = False
                dummy_reason = f"too_many_features frac={frac:.6f}"
                Printer.yellow(
                    f"[Dummy FAIL] too many features ({non_zero_area}/{image_area}={frac*100:.2f}%) "
                    f"p={prob_thresh}, u={uncer_thresh}"
                )
                # If you want to *allow* this case, delete the next line.
                raise ValueError("Dummy mask test failed: mask too dense.")

            # Dummy passed: apply mask generator
            dataset.set_mask_generator(mask_gen)

            # ✅ ONLY NOW start W&B (and only in optuna mode)
            if optuna_mode:
                wandb_run = wandb.init(
                    project="pixerv2_vo_optuna",
                    entity="droneslab",
                    name=exp_name,
                    group="optuna",
                    job_type="trial",
                    reinit=True,
                    config={
                        "feature_name": feature_name,
                        "feature_num": feature_num,
                        "prob_thresh": prob_thresh,
                        "uncer_thresh": uncer_thresh,
                        "is_baseline": is_baseline,
                    },
                )

                log_img = make_1x3_grid(img_t, mask)
                wandb.log({
                    "qual/dummy_grid": wandb.Image(
                        log_img,
                        caption=f"Dummy mask p={prob_thresh:.3f}, u={uncer_thresh:.3f}, frac={frac:.3%}"
                    )
                })

        else:
            # baseline: no mask_gen
            pass

        groundtruth = groundtruth_factory(config.dataset_settings)

        cam = PinholeCamera(config)

        # num_features = 2000  # how many features do you want to detect and track?
        num_features = feature_num
        # if (
        #     config.num_features_to_extract > 0
        # ):  # override the number of features to extract if we set something in the settings file
        #     num_features = config.num_features_to_extract

        # select your tracker configuration (see the file feature_tracker_configs.py)
        # LK_SHI_TOMASI, LK_FAST
        # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, LIGHTGLUE, XFEAT, XFEAT_XFEAT, LOFTR

        if feature_type == FeatureTrackerConfigs.LK_SHI_TOMASI:
            tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI
            tracker_config["num_features"] = num_features
        else:
            tracker_config = feature_type
            tracker_config["num_features"] = num_features

        feature_tracker = feature_tracker_factory(**tracker_config)

        if save_intermediate:
            save_loc = kResultsFolder
        else:
            save_loc = None

        # create visual odometry object
        if dataset.sensor_type == SensorType.RGBD:
            vo = VisualOdometryRgbdTensor(cam, groundtruth)  # only for RGBD
            Printer.green("Using VisualOdometryRgbdTensor")
        else:
            vo = VisualOdometryEducational(cam, groundtruth, feature_tracker)
            Printer.green("Using VisualOdometryEducational")
        time.sleep(1)  # time to read the message

        is_draw_traj_img = True
        traj_img_size = 800
        traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
        half_traj_img_size = int(0.5 * traj_img_size)
        draw_scale = 1

        plt3d = None

        viewer3D = None

        is_draw_3d = True
        is_draw_with_rerun = kUseRerun
        
        matched_points_plt = None
        err_plt = None
        if kVisualize:
            if is_draw_with_rerun:
                Rerun.init_vo()
            else:
                if kUsePangolin:
                    # viewer3D = Viewer3D(scale=dataset.scale_viewer_3d * 10)
                    pass
                else:
                    plt3d = Mplot3d(title="3D trajectory")

            is_draw_err = True
            err_plt = factory_plot2d(xlabel="img id", ylabel="m", title="error")

            is_draw_matched_points = True
            matched_points_plt = factory_plot2d(xlabel="img id", ylabel="# matches", title="# matches")

        img_id = 0

        # Lists for processing
        matched_kps = []
        num_inliers = []
        px_shifts = []
        rs = []
        ts = []
        kps = []
        des = []
        xs = []
        ys = []
        zs = []
        gtxs = []
        gtys = []
        gtzs = []
        img_id = 0
        images = []
        original_kps = []
        masked_kps = []
        est_times = []
        evo_cands = []
        evo_gts = []

        loop_start_time = time.perf_counter()
        while True:
            if img_id >= max_images-1 and max_images > 0:
                break
            img = None
            mask = None

            

            if dataset.is_ok:
                timestamp = dataset.getTimestamp()  # get current timestamp
                # img = dataset.getImageColor(img_id)
                img, mask = dataset.getImageColorAndMask(img_id)
                depth = dataset.getDepth(img_id)
                img_right = (
                    dataset.getImageColorRight(img_id)
                    if dataset.sensor_type == SensorType.STEREO
                    else None
                )

            if img is not None:
                if img_id == 1:
                    cv2.imwrite("silk_data/frame.png", img)
                matched_kp, num_inlier, px_shift, kp_cur, des_cur,rot,trans, kp_before, kp_after, est_time = vo.track(img, img_right, depth, img_id, timestamp, vo_mask=mask)  # main VO function
                if matched_kp is not None:
                    images.append(img)
                    matched_kps.append(matched_kp)
                    num_inliers.append(num_inlier)
                    px_shifts.append(px_shift)
                    kps.append(kp_cur)
                    des.append(des_cur)
                    rs.append(rot)
                    ts.append(trans)
                    original_kps.append(kp_before)
                    masked_kps.append(kp_after)
                    est_times.append(est_time)

                if (
                    len(vo.traj3d_est) > 1
                ):  # start drawing from the third image (when everything is initialized and flows in a normal way)

                    x, y, z = vo.traj3d_est[-1]
                    gt_x, gt_y, gt_z = vo.traj3d_gt[-1]

                    est = vo.evo_cand[-1]
                    gt = vo.evo_gt[-1]

                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    gtxs.append(gt_x)
                    gtys.append(gt_y)
                    gtzs.append(gt_z)

                    evo_cands.append(est)
                    evo_gts.append(gt)

                    if kVisualize:

                        if is_draw_traj_img:  # draw 2D trajectory (on the plane xz)
                            draw_x, draw_y = int(
                                draw_scale * x
                            ) + half_traj_img_size, half_traj_img_size - int(draw_scale * z)
                            draw_gt_x, draw_gt_y = int(
                                draw_scale * gt_x
                            ) + half_traj_img_size, half_traj_img_size - int(draw_scale * gt_z)
                            cv2.circle(
                                traj_img,
                                (draw_x, draw_y),
                                1,
                                (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0),
                                1,
                            )  # estimated from green to blue
                            cv2.circle(
                                traj_img, (draw_gt_x, draw_gt_y), 1, (0, 0, 255), 1
                            )  # groundtruth in red
                            # write text on traj_img
                            cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                            text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                            cv2.putText(
                                traj_img,
                                text,
                                (20, 40),
                                cv2.FONT_HERSHEY_PLAIN,
                                1,
                                (255, 255, 255),
                                1,
                                8,
                            )
                            # show

                            if is_draw_with_rerun:
                                Rerun.log_img_seq("trajectory_img/2d", img_id, traj_img)
                            else:
                                cv2.imshow("Trajectory", traj_img)

                        if is_draw_with_rerun:
                            Rerun.log_2d_seq_scalar("trajectory_error/err_x", img_id, math.fabs(gt_x - x))
                            Rerun.log_2d_seq_scalar("trajectory_error/err_y", img_id, math.fabs(gt_y - y))
                            Rerun.log_2d_seq_scalar("trajectory_error/err_z", img_id, math.fabs(gt_z - z))

                            Rerun.log_2d_seq_scalar(
                                "trajectory_stats/num_matches", img_id, vo.num_matched_kps
                            )
                            Rerun.log_2d_seq_scalar("trajectory_stats/num_inliers", img_id, vo.num_inliers)

                            Rerun.log_3d_camera_img_seq(img_id, vo.draw_img, None, cam, vo.poses[-1])
                            Rerun.log_3d_trajectory(img_id, vo.traj3d_est, "estimated", color=[0, 0, 255])
                            Rerun.log_3d_trajectory(img_id, vo.traj3d_gt, "ground_truth", color=[255, 0, 0])
                        else:
                            if is_draw_3d:  # draw 3d trajectory
                                plt3d.draw(vo.traj3d_gt, "ground truth", color="r", marker=".")
                                plt3d.draw(vo.traj3d_est, "estimated", color="g", marker=".")

                            if is_draw_err:  # draw error signals
                                errx = [img_id, math.fabs(gt_x - x)]
                                erry = [img_id, math.fabs(gt_y - y)]
                                errz = [img_id, math.fabs(gt_z - z)]
                                err_plt.draw(errx, "err_x", color="g")
                                err_plt.draw(erry, "err_y", color="b")
                                err_plt.draw(errz, "err_z", color="r")

                            if is_draw_matched_points:
                                matched_kps_signal = [img_id, vo.num_matched_kps]
                                inliers_signal = [img_id, vo.num_inliers]
                                matched_points_plt.draw(matched_kps_signal, "# matches", color="b")
                                matched_points_plt.draw(inliers_signal, "# inliers", color="g")

                # draw camera image
                if not is_draw_with_rerun and kVisualize:
                    cv2.imshow("Camera", vo.draw_img)

            else:
                print("End of dataset reached or error in reading data.")
                break

            # get keys
            key = matched_points_plt.get_key() if matched_points_plt is not None else None
            if key == "" or key is None:
                key = err_plt.get_key() if err_plt is not None else None
            if key == "" or key is None:
                key = plt3d.get_key() if plt3d is not None else None

            # press 'q' to exit!
            if kVisualize:
                key_cv = cv2.waitKey(1) & 0xFF
                if key == "q" or (key_cv == ord("q")):
                    break
                if viewer3D and viewer3D.is_closed():
                    break
            img_id += 1

        # print('press a key in order to exit...')
        # cv2.waitKey(0)
        if kVisualize:
            if is_draw_traj_img:
                if not os.path.exists(kResultsFolder):
                    os.makedirs(kResultsFolder, exist_ok=True)
                print(f"saving {kResultsFolder}/map.png")
                cv2.imwrite(f"{kResultsFolder}/map.png", traj_img)
            if plt3d:
                plt3d.quit()
            if viewer3D:
                viewer3D.quit()
            if err_plt:
                err_plt.quit()
            if matched_points_plt:
                matched_points_plt.quit()

            cv2.destroyAllWindows()
        
        loop_end_time = time.perf_counter()
        total_loop_time = loop_end_time - loop_start_time

        results = {
            'exp_name': exp_name,
            'feature_name': feature_name,
            'matched_kps': matched_kps,
            'num_inliers': num_inliers,
            'px_shifts': px_shifts,
            'rs': rs,
            'ts': ts,
            'kps': kps,
            'des': des,
            'xs': xs,
            'ys': ys,
            'zs': zs,
            'gtxs': gtxs,
            'gtys': gtys,
            'gtzs': gtzs,
            'original_kps': original_kps,
            'masked_kps': masked_kps,
            'est_times': est_times,
            'evo_cands': evo_cands,
            'evo_gts': evo_gts,
            'total_loop_time': total_loop_time,
            'prob_thresh': prob_thresh,
            'uncer_thresh': uncer_thresh,
        }

        original_kps = np.around(results['matched_kps'], decimals=2)
        masked_kps = np.around(results['num_inliers'], decimals=2)
        kp_reduction = (1 - (np.array(masked_kps) / np.array(original_kps))) * 100.0
        est_time = np.around(results['est_times'], decimals=4)
        loop_time = np.around(results['total_loop_time'], decimals=2)

        # wandb.log({
        #     "avg_original_kps": original_kps,
        #     "avg_masked_kps": masked_kps,
        #     "kp_reduction_%": kp_reduction,
        #     "avg_est_time_per_frame": est_time,
        #     "total_loop_time": loop_time,
        # })
        if optuna_mode and wandb_run is not None:
            wandb.log({'base_rmse': base_rmse})
            wandb.log({
                "avg_original_kps": np.mean(original_kps)})
            wandb.log({
                "avg_masked_kps": np.mean(masked_kps)})
            wandb.log({
                "kp_reduction_%": np.mean(kp_reduction)})
            wandb.log({
                "avg_est_time_per_frame": np.mean(est_time)})
            wandb.log({
                "total_loop_time": loop_time})

        # process_data(results, images=images, masks=images, draw_tracks=plot_tracks, plot_traj=plot_traj, traj_skips=20)
        run_folder = os.path.join(kResultsFolder, exp_name)
        os.makedirs(run_folder, exist_ok=True)

        # Always write the KITTI trajectories needed by evo_ape
        est_path = f"{run_folder}/{exp_name}_evo_cands.txt"
        gt_path  = f"{run_folder}/{exp_name}_evo_gts.txt"
        np.savetxt(est_path, np.array(results["evo_cands"]), fmt="%.9f")
        np.savetxt(gt_path,  np.array(results["evo_gts"]),  fmt="%.9f")

        # Compute evo RMSE (this is your Optuna objective)
        rmse = evo_ape_rmse_kitti(est_path, gt_path, monocular=True)

        if optuna_mode:
            wandb.log({"rmse_diff": base_rmse - rmse})

        # If NOT in optuna mode, keep your existing processing/reporting
        if not optuna_mode:
            process_data(
                results,
                images=images,
                masks=images,
                save_evo_report=save_evo_report,
                save_pkl=save_pkl,
                draw_tracks=plot_tracks,
                plot_traj=plot_traj,
                traj_skips=20,
            )
        if optuna_mode:
            wandb.log({"evo_ape_rmse": rmse})

        if wandb_run is not None:
            fig = make_xz_traj_figure(results["evo_cands"], results["evo_gts"], title=exp_name)
            wandb.log({"trajectory_xz": wandb.Image(fig)})
            plt.close(fig)
    except Exception as e:
        Printer.red(f"[Error] Experiment {exp_name} failed with error: {e}")
        traceback.print_exc()
        rmse = float('inf')
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return rmse

    

if __name__ == "__main__":

    max_images = 800

    # LK_SHI_TOMASI,LK_FAST, ORB, SIFT, AKAZE SHI_TOMASI_ORB, SHI_TOMASI_FREAK, FAST_ORB, FAST_FREAK, ORB2, BRISK, BRISK_TFEAT, KAZE, ROOT_SIFT, SUPERPOINT, XFEAT, XFEAT_XFEAT, XFEAT_LIGHTGLUE, LIGHTGLUE, LIGHTGLUE_DISK, LIGHTGLUE_ALIKED, LIGHTGLUESIFT, DELF, D2NET, R2D2, LFNET, CONTEXTDESC, KEYNET, DISK, ALIKED, KEYNETAFFNETHARDNET, ORB2_FREAK, ORB2_BEBLID, ORB2_HARDNET, ORB2_SOSNET, ORB2_L2NET

    FEATURE_TRACKER_PRESETS = [
        ("LK_SHI_TOMASI", FeatureTrackerConfigs.LK_SHI_TOMASI),
        ("LK_FAST", FeatureTrackerConfigs.LK_FAST),
        ("ORB", FeatureTrackerConfigs.ORB),
        ("SIFT", FeatureTrackerConfigs.SIFT),
        ("AKAZE", FeatureTrackerConfigs.AKAZE),
        ("SHI_TOMASI_ORB", FeatureTrackerConfigs.SHI_TOMASI_ORB),
        ("SHI_TOMASI_FREAK", FeatureTrackerConfigs.SHI_TOMASI_FREAK),
        ("FAST_ORB", FeatureTrackerConfigs.FAST_ORB),
        ("FAST_FREAK", FeatureTrackerConfigs.FAST_FREAK),
        ("ORB2", FeatureTrackerConfigs.ORB2),
        ("BRISK", FeatureTrackerConfigs.BRISK),
        ("BRISK_TFEAT", FeatureTrackerConfigs.BRISK_TFEAT),
        # ("KAZE", FeatureTrackerConfigs.KAZE), Dont use
        ("ROOT_SIFT", FeatureTrackerConfigs.ROOT_SIFT),
        ("SUPERPOINT", FeatureTrackerConfigs.SUPERPOINT),
        # ("XFEAT", FeatureTrackerConfigs.XFEAT), Dont use
        # ("XFEAT_XFEAT", FeatureTrackerConfigs.XFEAT_XFEAT), Dont use
        # ("XFEAT_LIGHTGLUE", FeatureTrackerConfigs.XFEAT_LIGHTGLUE), Dont use
        ("LIGHTGLUE", FeatureTrackerConfigs.LIGHTGLUE),
        # ("LIGHTGLUE_DISK", FeatureTrackerConfigs.LIGHTGLUE_DISK), Dont use
        ("LIGHTGLUE_ALIKED", FeatureTrackerConfigs.LIGHTGLUE_ALIKED),
        # ("LIGHTGLUESIFT", FeatureTrackerConfigs.LIGHTGLUESIFT),Dont use
        # ("DELF", FeatureTrackerConfigs.DELF), Dont use
        # ("D2NET", FeatureTrackerConfigs.D2NET), Error
        # ("R2D2", FeatureTrackerConfigs.R2D2), Error
        # ("LFNET", FeatureTrackerConfigs.LFNET), Dont use
        # ("CONTEXTDESC", FeatureTrackerConfigs.CONTEXTDESC), Dont use
        # ("KEYNET", FeatureTrackerConfigs.KEYNET), Dont use
        # ("DISK", FeatureTrackerConfigs.DISK), Dont use
        ("ALIKED", FeatureTrackerConfigs.ALIKED),
        # ("KEYNETAFFNETHARDNET", FeatureTrackerConfigs.KEYNETAFFNETHARDNET), Dont use
        ("ORB2_FREAK", FeatureTrackerConfigs.ORB2_FREAK),
        ("ORB2_BEBLID", FeatureTrackerConfigs.ORB2_BEBLID),
        ("ORB2_HARDNET", FeatureTrackerConfigs.ORB2_HARDNET),
        ("ORB2_SOSNET", FeatureTrackerConfigs.ORB2_SOSNET),
        ("ORB2_L2NET", FeatureTrackerConfigs.ORB2_L2NET),

    ]

    max_features = [100,400,500, 1000, 2000]
    base_lines = [False]

    probs = np.arange(0.0,1.01,0.1)
    uncers = np.arange(0.0,1.01,0.1)

    from itertools import product
    import traceback
    experiments = list(product(FEATURE_TRACKER_PRESETS, max_features, base_lines, probs, uncers))
    baselines = list(product(FEATURE_TRACKER_PRESETS, max_features, [True], [0.0], [0.0]))

    experiments = baselines + experiments

    import random
    import gc
    random.shuffle(experiments)

    # for (feature_name, feature_type), max_f, is_baseline in experiments:
    for (feature_name, feature_type), max_f, is_baseline, prob_thresh, uncer_thresh in experiments:
        exp_name = f'{"baseline" if is_baseline else "masked"}_${max_f}$_#{feature_name}#_*{prob_thresh:.2f}*_<{uncer_thresh:.2f}>'
        try:
            run_exp(
                exp_name=exp_name,
                feature_type=feature_type,
                feature_name=feature_name,
                is_baseline=is_baseline,
                feature_num=max_f,
                max_images=max_images,
                save_intermediate=False,
                plot_tracks=False,
                plot_traj=True,
                prob_thresh=prob_thresh,
                uncer_thresh=uncer_thresh,
            )

            # garbage collection and torch cache clearing
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Experiment {exp_name} failed with exception: {e}")
            tb = traceback.format_exc()
            with open(f'{kResultsFolder}/{exp_name}_error.txt', 'w') as f:
                f.write(tb)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    # run_exp(
    #     exp_name="masked",
    #     feature_type = FeatureTrackerConfigs.LK_SHI_TOMASI,
    #     feature_name = "LK_SHI_TOMASI",
    #     feature_num = 2000,
    #     max_images = 1000,
    #     save_intermediate = True,
    #     plot_tracks = False,
    #     plot_traj = True,
    # )
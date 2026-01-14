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
import matplotlib.pyplot as plt
from pyslam.config import Config

from pyslam.slam.visual_odometry import VisualOdometryEducational
from pyslam.slam.visual_odometry_rgbd import (
    VisualOdometryRgbd,
    VisualOdometryRgbdTensor,
)
from pyslam.slam.camera import PinholeCamera

from pyslam.io.ground_truth import groundtruth_factory
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import DatasetType, SensorType

from pyslam.viz.mplot_thread import Mplot2d, Mplot3d
# from pyslam.viz.qplot_thread import Qplot2d
from pyslam.viz.rerun_interface import Rerun

from pyslam.local_features.feature_tracker import (
    feature_tracker_factory,
    FeatureTrackerTypes,
)
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.utilities.utils_sys import Printer

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

def factory_plot2d(*args, **kwargs):
    if kVisualize:
        if kUseRerun:
            return None
        else:
            return Mplot2d(*args, **kwargs)
    else:
        return None
    
def run_evo(cmd, cwd=None):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(cmd, check=True, cwd=cwd, env=env)
    
def generate_evo_report(evo_cand, evo_gt, out_dir="evo_report", monocular=False, rpe_delta=1, orientation="xz"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    est = evo_cand
    gt = evo_gt
    
    # common flags
    align = ["-a"]
    if monocular:
        align += ["--correct_scale"]

    # APE
    run_evo([
        "evo_ape", "kitti", str(gt), str(est),
        *align,
        "--save_results", str(out_dir / "ape.zip"),
        "--save_plot", str(out_dir / "ape.png"),
    ])

    # RPE (per-frame by default; can change delta)
    run_evo([
        "evo_rpe", "kitti", str(gt), str(est),
        *align,
        "--delta", str(rpe_delta),
        "--save_results", str(out_dir / "rpe.zip"),
        "--save_plot", str(out_dir / "rpe.png"),
    ])

    # Trajectory overlay (saved only)
    run_evo([
        "evo_traj", "kitti", str(gt), str(est),
        "--plot_mode", orientation,
        "--save_plot", str(out_dir / f"traj_{orientation}.png"),
    ])

    # Also save stdout tables (nice for quick diff/grep)
    (out_dir / "ape.txt").write_text(
        subprocess.check_output(
            ["evo_ape", "kitti", str(gt), str(est), *align],
            env={**os.environ, "MPLBACKEND": "Agg"},
            text=True
        )
    )
    (out_dir / "rpe.txt").write_text(
        subprocess.check_output(
            ["evo_rpe", "kitti", str(gt), str(est), *align,
             "--delta", str(rpe_delta)],
            env={**os.environ, "MPLBACKEND": "Agg"},
            text=True
        )
    )

    return out_dir


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
        feature_num: int = 2000,
        max_images: int = -1,
        save_intermediate: bool = False,
        save_pkl: bool = True,
        save_evo_report: bool = True,
        plot_tracks: bool = False,
        plot_traj: bool = False,
        ):
    
    if os.environ.get('PYSLAM_CONFIG') is None:
        raise RuntimeError("Please set the PYSLAM_CONFIG with the same scope as this script!")
    
    config_loc = os.environ.get('PYSLAM_CONFIG')
    config_name = config_loc.split('.')[0]

    if exp_name == "":
        exp_name = f'{config_name}_#{feature_name}#_@{feature_num}@' 
    else:
        exp_name = f'{exp_name}_#{feature_name}#_@{feature_num}@'
    
    config = Config()

    dataset = dataset_factory(config)

    groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(config)

    # num_features = 2000  # how many features do you want to detect and track?
    num_features = feature_num
    if (
        config.num_features_to_extract > 0
    ):  # override the number of features to extract if we set something in the settings file
        num_features = config.num_features_to_extract

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
    }



    # process_data(results, images=images, masks=images, draw_tracks=plot_tracks, plot_traj=plot_traj, traj_skips=20)
    process_data(
        results, 
        images=images, 
        masks=images, 
        save_evo_report=save_evo_report,
        save_pkl=save_pkl,
        draw_tracks=plot_tracks, 
        plot_traj=plot_traj, 
        traj_skips=20
    )

if __name__ == "__main__":

    run_exp(
        exp_name="masked",
        feature_type = FeatureTrackerConfigs.LK_SHI_TOMASI,
        feature_name = "LK_SHI_TOMASI",
        feature_num = 2000,
        max_images = 1000,
        save_intermediate = True,
        plot_tracks = False,
        plot_traj = True,
    )
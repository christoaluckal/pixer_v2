import optuna
import math
import gc
import torch
from main_vo import run_exp
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
import json
from pathlib import Path
import wandb

import optuna
import math
import gc
import torch
import wandb
from main_vo import run_exp
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
import traceback
import numpy as np
import cv2

def to_uint8_rgb(img):
    """
    Convert various possible image containers to a numpy uint8 RGB image (H,W,3).
    Supports:
      - np.ndarray (H,W), (H,W,3), float or uint8
      - wandb.Image (best-effort: uses .image if available)
    Returns None if conversion fails.
    """
    if img is None:
        return None

    # If it's a wandb.Image, try to extract underlying numpy array
    if isinstance(img, wandb.Image):
        # Newer wandb often exposes `.image` as numpy array or PIL image
        x = getattr(img, "image", None)
        if x is None:
            return None
        img = x

    # If it's a PIL image
    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
    except Exception:
        pass

    if not isinstance(img, np.ndarray):
        return None

    # Ensure 3 channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        return None

    # Drop alpha if present
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # Convert dtype to uint8
    if img.dtype == np.uint8:
        return img

    # float images could be 0..1 or 0..255; normalize safely
    img_f = img.astype(np.float32)
    mx = float(np.nanmax(img_f)) if img_f.size else 0.0
    if mx <= 1.0:
        img_f = img_f * 255.0
    img_u8 = np.clip(img_f, 0, 255).astype(np.uint8)
    img_u8 = np.ascontiguousarray(img_u8)
    return img_u8

def add_label(img_u8, text, org=(10, 35)):
    if img_u8 is None:
        return None

    # Ensure numpy + uint8
    if not isinstance(img_u8, np.ndarray):
        raise TypeError(f"add_label expected np.ndarray, got {type(img_u8)}")
    if img_u8.dtype != np.uint8:
        img_u8 = img_u8.astype(np.uint8)

    # **CRITICAL**: OpenCV requires C-contiguous memory
    if not img_u8.flags["C_CONTIGUOUS"]:
        img_u8 = np.ascontiguousarray(img_u8)

    # Also ensure writable (rare, but can happen)
    if not img_u8.flags["WRITEABLE"]:
        img_u8 = img_u8.copy()

    cv2.putText(img_u8, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img_u8, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1, cv2.LINE_AA)
    return img_u8



def hstack_images(imgs, pad=10, pad_value=255):
    """
    imgs: list of np.ndarray (H, W, 3)
    """
    imgs = [im for im in imgs if im is not None]
    if len(imgs) == 0:
        return None

    # normalize heights
    h_max = max(im.shape[0] for im in imgs)
    resized = []
    for im in imgs:
        if im.shape[0] != h_max:
            scale = h_max / im.shape[0]
            new_w = int(im.shape[1] * scale)
            im = cv2.resize(im, (new_w, h_max))
        resized.append(im)

    if pad > 0:
        pad_img = pad_value * np.ones((h_max, pad, 3), dtype=np.uint8)
        out = resized[0]
        for im in resized[1:]:
            out = np.hstack([out, pad_img, im])
    else:
        out = np.hstack(resized)

    return out

class OptunaWrapper:
    def __init__(self, tracker_name: str, feature_num: int, max_images: int = 800):
        self.tracker_name = tracker_name
        self.feature_num = feature_num
        self.max_images = max_images

        self.tracker_map = {
            "LK_SHI_TOMASI": FeatureTrackerConfigs.LK_SHI_TOMASI,
            "ORB": FeatureTrackerConfigs.ORB,
            "SIFT": FeatureTrackerConfigs.SIFT,
            # "ORB2": FeatureTrackerConfigs.ORB2,
            "BRISK": FeatureTrackerConfigs.BRISK,
            "SUPERPOINT": FeatureTrackerConfigs.SUPERPOINT,
        }
        self.feature_type = self.tracker_map[self.tracker_name]

        self._baseline_rmse = None
        self._sota_rmse = None
        self._wandb_run = None  # holds the single run

    def run_baseline(self) -> float:
        if self._baseline_rmse is None:
            self._baseline_rmse, _ = run_exp(
                exp_name=f"baseline_{self.tracker_name}_F{self.feature_num}",
                feature_type=self.feature_type,
                feature_name=self.tracker_name,
                feature_num=self.feature_num,
                max_images=self.max_images,
                is_baseline=True,
                prob_thresh=0.0,
                uncer_thresh=0.0,
                save_evo_report=False,
                plot_traj=False,
                optuna_mode=False,
                mask_source="none"
            )
        return self._baseline_rmse

    def run_sota(self) -> float:
        if self._sota_rmse is None:
            self._sota_rmse, _ = run_exp(
                exp_name=f"sivo_{self.tracker_name}_F{self.feature_num}",
                feature_type=self.feature_type,
                feature_name=self.tracker_name,
                feature_num=self.feature_num,
                max_images=self.max_images,
                is_baseline=False,
                prob_thresh=0.0,
                uncer_thresh=0.0,
                save_evo_report=False,
                plot_traj=False,
                optuna_mode=False,
                mask_source="sota_png"
            )
        return self._sota_rmse

    def objective(self, trial: optuna.Trial):
        prob_thresh = 0.0
        uncer_thresh = trial.suggest_float("uncer_thresh", 0.0, 1.0)

        exp_name = (
            f"optuna_{trial.number}_{self.tracker_name}_F{self.feature_num}"
            f"_p{prob_thresh:.3f}_u{uncer_thresh:.3f}"
        )

        try:
            # If you want baseline once, uncomment:
            rmse_base = self.run_baseline()
            # rmse_base = 0.0

            rmse_sivo = self.run_sota()

            rmse_masked, wandb_log_dict = run_exp(
                exp_name=exp_name,
                feature_type=self.feature_type,
                feature_name=self.tracker_name,
                feature_num=self.feature_num,
                max_images=self.max_images,
                is_baseline=False,
                save_evo_report=False,
                plot_traj=False,
                prob_thresh=prob_thresh,
                uncer_thresh=uncer_thresh,
                optuna_mode=True,
                base_rmse=rmse_base,
            )

            improvement = rmse_base - rmse_masked

            # log EVERYTHING to the same run, using trial.number as step
            log_payload = {
                "sivo_rmse": rmse_sivo,
                "prob_thresh": prob_thresh,
                "uncer_thresh": uncer_thresh,
                "rmse_base": wandb_log_dict.get("base_rmse", rmse_base),
                "rmse_masked": rmse_masked,
                "improvement": improvement,
                "avg_original_kps": wandb_log_dict["avg_original_kps"],
                "avg_masked_kps": wandb_log_dict["avg_masked_kps"],
                "kp_reduction_%": wandb_log_dict["kp_reduction_%"],
                "avg_est_time_per_frame": wandb_log_dict["avg_est_time_per_frame"],
                "total_loop_time": wandb_log_dict["total_loop_time"],
                "rmse_diff": wandb_log_dict["rmse_diff"],
                "evo_ape_rmse": wandb_log_dict["evo_ape_rmse"],
                "trial_failed": 0
            }

            # images (only if present)
            if "log_img" in wandb_log_dict and wandb_log_dict["log_img"] is not None:
                log_payload["log_img"] = wandb.Image(wandb_log_dict["log_img"])

            if wandb_log_dict["trajectory_xz"] is not None:
                log_payload["trajectory_xz"] = wandb_log_dict["trajectory_xz"]

            wandb.log(log_payload, step=trial.number)

            # store for Optuna
            trial.set_user_attr("rmse_base", rmse_base)
            trial.set_user_attr("rmse_sivo", rmse_sivo)
            trial.set_user_attr("rmse_masked", rmse_masked)
            trial.set_user_attr("improvement", improvement)

            return improvement

        except (ValueError,KeyError) as e:
            traceback.print_exc()
            for k, v in wandb_log_dict.items():
                print(k,v)
            # input()
            print(f"Error running trial: {e}")
            # optionally log the failure at this step
            wandb.log({"trial_failed": 1, "error": str(e)}, step=trial.number)
            return -math.inf

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run_optimization(self, n_trials: int = 100, sampler_seed: int = 0):
        # ONE run for this (tracker, feature_num)
        self._wandb_run = wandb.init(
            project="pixerv2_vo_optuna",
            entity="droneslab",
            name=f"{self.tracker_name}_F{self.feature_num}",  # e.g. ORB_F400
            group="optuna",
            job_type="study",
            config={
                "feature_name": self.tracker_name,
                "feature_num": self.feature_num,
                "max_images": self.max_images,
                "n_trials": n_trials,
                "sampler_seed": sampler_seed,
            },
            reinit=False,  # important: don't create new runs
        )

        try:
            # sampler optional
            # sampler = optuna.samplers.TPESampler(seed=sampler_seed)
            # study = optuna.create_study(direction="maximize", sampler=sampler)
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=n_trials)

            best_uncer = study.best_trial.params["uncer_thresh"]

            maximgs = 1500
            final_step = len(study.trials) 
            final1500 = self.final_eval_1500(best_uncer=best_uncer, final_images=maximgs, wandb_step=final_step)

            # Store in study user attrs so it ends up in your JSON record too
            study.set_user_attr(f"finallog", final1500)

            return study
        finally:
            wandb.finish()
            self._wandb_run = None
            self._baseline_rmse = None
            self._sota_rmse = None

    def final_eval_1500(self, best_uncer: float, final_images: int = 1500, wandb_step=0):
        # Run baseline @1500
        rmse_base_1500, wandb_base = run_exp(
            exp_name=f"final{final_images}_baseline_{self.tracker_name}_F{self.feature_num}",
            feature_type=self.feature_type,
            feature_name=self.tracker_name,
            feature_num=self.feature_num,
            max_images=final_images,
            is_baseline=True,
            prob_thresh=0.0,
            uncer_thresh=0.0,
            save_evo_report=True,   # final metric: you may want this ON
            plot_traj=True,
            optuna_mode=True,
            base_rmse=0.0,
            mask_source="none",
            run_process_data=True,
        )

        # input()

        # Run YOURS @1500 with best params (mask enabled)
        rmse_ours_1500, wandb_ours = run_exp(
            exp_name=f"final{final_images}_ours_{self.tracker_name}_F{self.feature_num}_u{best_uncer:.3f}",
            feature_type=self.feature_type,
            feature_name=self.tracker_name,
            feature_num=self.feature_num,
            max_images=final_images,
            is_baseline=False,
            prob_thresh=0.0,
            uncer_thresh=best_uncer,
            save_evo_report=True,
            plot_traj=True,
            optuna_mode=True,
            base_rmse=rmse_base_1500,
            mask_source="ours",
            run_process_data=True,
        )

        # input()

        # Run SOTA @1500 (precomputed PNG masks)
        # This assumes you have run_exp support something like mask_source="sota_png"
        rmse_sota_1500, wandb_sota = run_exp(
            exp_name=f"final{final_images}_sota_{self.tracker_name}_F{self.feature_num}",
            feature_type=self.feature_type,
            feature_name=self.tracker_name,
            feature_num=self.feature_num,
            max_images=final_images,
            is_baseline=False,
            prob_thresh=0.0,
            uncer_thresh=0.0,
            save_evo_report=True,
            plot_traj=True,
            optuna_mode=True,
            base_rmse=rmse_base_1500,
            mask_source="sota_png",     # <--- see note below
            run_process_data=True,
        )

        # Log final summary at step AFTER the last trial
        # step = None
        # try:
        #     step = len(wandb.run.history._data)  # not stable across wandb versions
        # except Exception:
        #     step = 999999  # fallback; not critical

        # wandb.log({
        #     f"final{final_images}/rmse_base": rmse_base_1500,
        #     f"final{final_images}/rmse_ours": rmse_ours_1500,
        #     f"final{final_images}/rmse_sota": rmse_sota_1500,
        #     f"final{final_images}/improvement_ours": rmse_base_1500 - rmse_ours_1500,
        #     f"final{final_images}/improvement_sota": rmse_base_1500 - rmse_sota_1500,
        #     f"final{final_images}/best_uncer_thresh": best_uncer,
        #     # Optional: also store some comparable stats if present
        #     f"final{final_images}/avg_original_kps_ours": wandb_ours.get("avg_original_kps", None),
        #     f"final{final_images}/avg_masked_kps_ours": wandb_ours.get("avg_masked_kps", None),
        #     f"final{final_images}/avg_kp_reduction_ours": wandb_ours["kp_reduction_%"],
        #     f"final{final_images}/avg_original_kps_sota": wandb_sota.get("avg_original_kps", None),
        #     f"final{final_images}/avg_masked_kps_sota": wandb_sota.get("avg_masked_kps", None),
        #     f"final{final_images}/avg_kp_reduction_sota": wandb_sota["kp_reduction_%"]
        # }, step=wandb_step)

        final_payload = {
            f"final{final_images}/rmse_base": rmse_base_1500,
            f"final{final_images}/rmse_ours": rmse_ours_1500,
            f"final{final_images}/rmse_sota": rmse_sota_1500,
            f"final{final_images}/improvement_ours": rmse_base_1500 - rmse_ours_1500,
            f"final{final_images}/improvement_sota": rmse_base_1500 - rmse_sota_1500,
            f"final{final_images}/best_uncer_thresh": best_uncer,

            # Optional: stats
            f"final{final_images}/avg_original_kps_ours": wandb_ours.get("avg_original_kps", None),
            f"final{final_images}/avg_masked_kps_ours": wandb_ours.get("avg_masked_kps", None),
            f"final{final_images}/avg_kp_reduction_ours": wandb_ours.get("kp_reduction_%", None),
            f"final{final_images}/avg_original_kps_sota": wandb_sota.get("avg_original_kps", None),
            f"final{final_images}/avg_masked_kps_sota": wandb_sota.get("avg_masked_kps", None),
            f"final{final_images}/avg_kp_reduction_sota": wandb_sota.get("kp_reduction_%", None),
        }

        # ---- trajectories (run_exp returns wandb.Image for trajectory_xz) ----
        # if wandb_base and wandb_base.get("trajectory_xz") is not None:
        #     final_payload[f"final{final_images}/trajectory_base_xz"] = wandb_base["trajectory_xz"]

        # if wandb_ours and wandb_ours.get("trajectory_xz") is not None:
        #     final_payload[f"final{final_images}/trajectory_ours_xz"] = wandb_ours["trajectory_xz"]

        # if wandb_sota and wandb_sota.get("trajectory_xz") is not None:
        #     final_payload[f"final{final_images}/trajectory_sota_xz"] = wandb_sota["trajectory_xz"]

        traj_base = to_uint8_rgb(wandb_base.get("trajectory_xz"))
        traj_ours = to_uint8_rgb(wandb_ours.get("trajectory_xz"))
        traj_sota = to_uint8_rgb(wandb_sota.get("trajectory_xz"))

        traj_base = add_label(traj_base, "Baseline")
        traj_ours = add_label(traj_ours, "Ours")
        traj_sota = add_label(traj_sota, "SOTA")

        traj_triptych = hstack_images(
            [traj_base, traj_ours, traj_sota],
            pad=12
        )

        if traj_triptych is not None:
            final_payload[f"final{final_images}/trajectory_xz_triptych"] = wandb.Image(
                traj_triptych,
                caption="Baseline | Ours | SOTA"
            )

        # ---- 1x3 dummy grids ONLY for ours + sota (run_exp returns raw image array) ----
        if wandb_ours and wandb_ours.get("log_img") is not None:
            final_payload[f"final{final_images}/dummy_grid_ours"] = wandb.Image(wandb_ours["log_img"])

        if wandb_sota and wandb_sota.get("log_img") is not None:
            final_payload[f"final{final_images}/dummy_grid_sota"] = wandb.Image(wandb_sota["log_img"])

        wandb.log(final_payload, step=wandb_step)


        return {
            f"rmse_base_{final_images}": rmse_base_1500,
            f"rmse_ours_{final_images}": rmse_ours_1500,
            f"rmse_sota_{final_images}": rmse_sota_1500,
            f"impr_ours_{final_images}": rmse_base_1500 - rmse_ours_1500,
            f"impr_sota_{final_images}": rmse_base_1500 - rmse_sota_1500,

            # ---- NEW: final reduction percentages ----
            f"kp_reduction_ours_{final_images}": wandb_ours.get("kp_reduction_%", None),
            f"kp_reduction_sota_{final_images}": wandb_sota.get("kp_reduction_%", None),
        }


def study_to_record(study, tracker_name: str, feature_num: int, max_images: int):
    t = study.best_trial
    rmse_base = t.user_attrs.get("rmse_base")
    rmse_sivo = t.user_attrs.get("rmse_sivo")
    rmse_masked = t.user_attrs.get("rmse_masked")
    improvement = t.value  # since objective is rmse_base - rmse_masked
    final_log = study.user_attrs.get("finallog", None)
    return {
        "tracker": tracker_name,
        "feature_num": feature_num,
        "max_images": max_images,
        "best": {
            "objective_improvement": improvement,
            "prob_thresh": 0.0,
            "uncer_thresh": t.params["uncer_thresh"],
            "rmse_base": rmse_base,
            "rmse_sivo": rmse_sivo,
            "rmse_masked": rmse_masked,
        },
        "optuna": {
            "best_trial_number": t.number,
            "n_trials": len(study.trials),
        },
        "finallog": final_log
    }

def run_suite(
    trackers: list[str],
    feature_nums: list[int],
    max_images: int = 800,
    n_trials: int = 100,
    out_path: str = "results/optuna_best_params.json",
):
    results = []

    for tracker_name in trackers:
        for feature_num in feature_nums:
            wrapper = OptunaWrapper(
                tracker_name=tracker_name,
                feature_num=feature_num,
                max_images=max_images,
            )
            study = wrapper.run_optimization(n_trials=n_trials, sampler_seed=0)

            rec = study_to_record(study, tracker_name, feature_num, max_images)
            results.append(rec)

            # write incrementally (so you don’t lose progress on crash)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

            print(f"[OK] {tracker_name} F={feature_num} best improvement={rec['best']['objective_improvement']:.6f}")

    return results

if __name__ == "__main__":
    trackers = [
        "LK_SHI_TOMASI",
        "ORB",
        "SIFT",
        "BRISK",
        "SUPERPOINT",
    ]
    feature_nums = [
        400, 
        1000, 
        2000, 
        3000
        ]

    run_suite(
        trackers=trackers,
        feature_nums=feature_nums,
        max_images=800,
        n_trials=200,
        out_path="results/optuna_best_params.json",
    )
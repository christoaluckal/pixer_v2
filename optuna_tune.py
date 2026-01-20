import optuna
import math
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from main_vo import run_exp
import wandb
import torch
import gc
baseline_cache = {}

def get_baseline_rmse(tracker_name, feature_type, feature_num, max_images):
    key = (tracker_name, feature_num, max_images)
    if key not in baseline_cache:
        baseline_cache[key] = run_exp(
            exp_name=f"baseline_{tracker_name}_F{feature_num}",
            feature_type=feature_type,
            feature_name=tracker_name,
            feature_num=feature_num,
            max_images=max_images,
            is_baseline=True,        # <- baseline path (no masking)
            prob_thresh=0.0,         # unused in baseline, but harmless
            uncer_thresh=0.0,
            save_evo_report=False,
            plot_traj=False,
            optuna_mode=False,       # <- IMPORTANT: prevents wandb logging in your run_exp
        )
    return baseline_cache[key]


def objective(trial):
    # sample parameters
    # feature_num = trial.suggest_int("feature_num", 200, 3000, step=200)
    feature_num = 3000

    prob_thresh = trial.suggest_float("prob_thresh", 0.0, 1.0)
    uncer_thresh = trial.suggest_float("uncer_thresh", 0.0, 1.0)

    # (optional) choose tracker preset
    tracker_name = trial.suggest_categorical(
        "tracker",
        # ["LK_SHI_TOMASI", "ORB", "SIFT", "AKAZE", "LIGHTGLUE", "ALIKED"],
        ["LK_SHI_TOMASI"],
    )
    tracker_map = {
        "LK_SHI_TOMASI": FeatureTrackerConfigs.LK_SHI_TOMASI,
        # "ORB": FeatureTrackerConfigs.ORB,
        # "SIFT": FeatureTrackerConfigs.SIFT,
        # "AKAZE": FeatureTrackerConfigs.AKAZE,
        # "LIGHTGLUE": FeatureTrackerConfigs.LIGHTGLUE,
        # "ALIKED": FeatureTrackerConfigs.ALIKED,
    }
    feature_type = tracker_map[tracker_name]

    exp_name = f"optuna_{trial.number}_{tracker_name}_F{feature_num}_p{prob_thresh:.3f}_u{uncer_thresh:.3f}"
    max_images = 800

    try:

        rmse_base = get_baseline_rmse(tracker_name, feature_type, feature_num, max_images)

        rmse = run_exp(
            exp_name=exp_name,
            feature_type=feature_type,
            feature_name=tracker_name,
            feature_num=feature_num,
            max_images=max_images,
            save_evo_report=False,
            plot_traj=False,          # if you want the plot
            prob_thresh=prob_thresh,
            uncer_thresh=uncer_thresh,
            optuna_mode=True,
            base_rmse=rmse_base,
            # add: trial_number=trial.number (optional)
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return rmse_base - rmse 
    except Exception:
        return -math.inf


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_params)

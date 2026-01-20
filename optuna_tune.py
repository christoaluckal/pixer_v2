import optuna
import math
import gc
import torch
from main_vo import run_exp
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
import json
from pathlib import Path

class OptunaWrapper:
    def __init__(self, tracker_name: str, feature_num: int, max_images: int = 800):
        self.tracker_name = tracker_name
        self.feature_num = feature_num
        self.max_images = max_images

        self.tracker_map = {
            "LK_SHI_TOMASI": FeatureTrackerConfigs.LK_SHI_TOMASI,
            "ORB": FeatureTrackerConfigs.ORB,
            "SIFT": FeatureTrackerConfigs.SIFT,
            "ORB2": FeatureTrackerConfigs.ORB2,
            "BRISK": FeatureTrackerConfigs.BRISK,
            "SUPERPOINT": FeatureTrackerConfigs.SUPERPOINT,
        }
        self.feature_type = self.tracker_map[self.tracker_name]

        self._baseline_rmse = None

    def run_baseline(self) -> float:
        if self._baseline_rmse is None:
            self._baseline_rmse = run_exp(
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
            )
        return self._baseline_rmse

    def objective(self, trial):
        prob_thresh = trial.suggest_float("prob_thresh", 0.0, 1.0)
        uncer_thresh = trial.suggest_float("uncer_thresh", 0.0, 1.0)

        exp_name = (
            f"optuna_{trial.number}_{self.tracker_name}_F{self.feature_num}"
            f"_p{prob_thresh:.3f}_u{uncer_thresh:.3f}"
        )

        try:
            # rmse_base = self.run_baseline()
            rmse_base = 0.0
            rmse_masked = run_exp(
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
                optuna_mode=True,   # only successful trials should create wandb runs (per your logic)
                base_rmse=rmse_base,  # if your run_exp uses it
            )

            # optional: store extra info for later export
            trial.set_user_attr("rmse_base", rmse_base)
            trial.set_user_attr("rmse_masked", rmse_masked)
            trial.set_user_attr("improvement", rmse_base - rmse_masked)

            return rmse_base - rmse_masked  # maximize improvement

        except Exception:
            return -math.inf

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run_optimization(self, n_trials: int = 100, sampler_seed: int = 0):
        sampler = optuna.samplers.TPESampler(seed=sampler_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self.objective, n_trials=n_trials)
        return study


def study_to_record(study, tracker_name: str, feature_num: int, max_images: int):
    t = study.best_trial
    rmse_base = t.user_attrs.get("rmse_base")
    rmse_masked = t.user_attrs.get("rmse_masked")
    improvement = t.value  # since objective is rmse_base - rmse_masked

    return {
        "tracker": tracker_name,
        "feature_num": feature_num,
        "max_images": max_images,
        "best": {
            "objective_improvement": improvement,
            "prob_thresh": t.params["prob_thresh"],
            "uncer_thresh": t.params["uncer_thresh"],
            "rmse_base": rmse_base,
            "rmse_masked": rmse_masked,
        },
        "optuna": {
            "best_trial_number": t.number,
            "n_trials": len(study.trials),
        },
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
        "ORB2",
        "BRISK",
        "SUPERPOINT",
    ]
    feature_nums = [400, 1000, 2000, 3000]

    run_suite(
        trackers=trackers,
        feature_nums=feature_nums,
        max_images=800,
        n_trials=50,
        out_path="results/optuna_best_params.json",
    )
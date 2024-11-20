import os
import json
import math
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

from src.common.utils import print_line, clear_line, get_ci
from src.evaluation.utils import get_sample_trajectories
from src.evaluation.base_eval import Base_Eval, Base_Collate


class Eval_Reward_Collate(Base_Collate):
    def _get_eval_function_results(self, save_loc: str):
        save_path = f"{save_loc}/reward/reward_stats.json"
        if not os.path.isfile(save_path):
            return None

        with open(save_path, "r") as results_file:
            return json.load(results_file)

    def _collate_results(self, all_results):
        for metric in ["mean", "iqm", "lqm", "uqm", "l10m", "u10m", "min", "max"]:
            self._collate_metric(all_results, metric)

        self._collate_performance_profile(all_results)

    def _collate_metric(self, all_results, metric: str):
        n_envs = len(self.env_list)
        fig, axs = plt.subplots(1, n_envs, figsize=(max(n_envs, 3), 3), sharey=True)
        cmap = plt.get_cmap("tab10")
        if n_envs == 1:
            axs = np.array([axs])

        registered_agent_names = []
        for env_i, (env_name, env_values) in enumerate(all_results.items()):
            axs[env_i].set_title(
                "".join([en[0].capitalize() for en in env_name.split("_")[:2]])
            )
            env_agent_names = []
            x_ticks = []
            for agent_i, (agent_name, agent_values) in enumerate(env_values.items()):
                all_metric_values = []
                agent_name: str = agent_name.replace("_", " ")
                for seed_data in agent_values:
                    for ckpt_name, ckpt_data in seed_data.items():
                        if ckpt_data is not None and metric in ckpt_data:
                            all_metric_values.append(ckpt_data[metric])

                all_metric_values = np.array(all_metric_values)

                if all_metric_values.shape[0] == 0:
                    continue

                if agent_name not in registered_agent_names:
                    registered_agent_names.append(agent_name)

                metric_mean = all_metric_values.mean(axis=0)
                lower_ci, upper_ci = get_ci(all_metric_values)

                x_vals = [agent_i - 0.4, agent_i + 0.4]
                x_ticks.append(agent_i)
                axs[env_i].hlines(metric_mean, x_vals[0], x_vals[1], color="black")
                axs[env_i].fill_between(
                    x_vals,
                    [lower_ci] * 2,
                    [upper_ci] * 2,
                    color=cmap(registered_agent_names.index(agent_name)),
                    alpha=0.5,
                )
                env_agent_names.append(agent_name)
            axs[env_i].set_xticks(x_ticks)
            axs[env_i].set_xticklabels(env_agent_names, rotation=90)
            if len(x_ticks) != 0:
                axs[env_i].scatter([x_ticks[0]] * 2, [0, 1000], s=0)

        fig.suptitle(f"{metric} Mean \u00B1 95% CI")
        fig.tight_layout()
        save_dir = f"{self.results_dir}/collated_results/final/reward"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{metric}.eps", format="eps")
        plt.close()

    def _collate_performance_profile(self, all_results):
        registered_agent_names = []
        for env_name, env_values in all_results.items():
            print_line(env_name)

            fig, axs = plt.subplots(figsize=(6, 6))
            cmap = plt.get_cmap("tab10")
            axs.set_title("".join(" ".join(env_name.split("_")[:2])))

            agent_raw_values = {}
            min_metric, max_metric = np.inf, -np.inf
            for agent_name, agent_values in env_values.items():
                all_raw_values = []
                agent_name: str = agent_name.replace("_", " ")
                for seed_data in agent_values:
                    for ckpt_data in seed_data.values():
                        if ckpt_data is not None and "raw" in ckpt_data:
                            all_raw_values.append(ckpt_data["raw"])
                            min_metric = min(min_metric, min(ckpt_data["raw"]))
                            max_metric = max(max_metric, max(ckpt_data["raw"]))

                if len(all_raw_values) == 0:
                    continue

                agent_raw_values[agent_name] = np.stack(all_raw_values, axis=-1)

                if agent_name not in registered_agent_names:
                    registered_agent_names.append(agent_name)

            tau_vals = np.arange(
                max(math.floor(min_metric * 0.9), 0),
                min(math.ceil(max_metric * 1.1), 1000),
                1,
            )
            score_distributions, score_distributions_cis = (
                rly.create_performance_profile(agent_raw_values, tau_vals)
            )

            plot_utils.plot_performance_profiles(
                score_distributions,
                tau_vals,
                performance_profile_cis=score_distributions_cis,
                ax=axs,
            )
            axs.legend()
            fig.tight_layout()
            save_dir = (
                f"{self.results_dir}/collated_results/final/reward/performance_profiles"
            )
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{env_name}.png")
            plt.close()

    def _collate_series_results(self, all_results):
        raise NotImplementedError()


class Eval_Reward(Base_Eval):
    @classmethod
    def get_collate_results_class(cls) -> Union[Base_Collate, None]:
        return Eval_Reward_Collate

    @classmethod
    def get_eval_name(self):
        return "reward"

    def _plt_all_figs(self, save_loc: str) -> None:
        pass

    def _save_metrics(self, save_loc: str) -> None:
        self._eval_reward(save_loc, n_inits=800)

    def _save_reward(
        self, reward_mean: np.ndarray, reward_std: np.ndarray, save_loc: str
    ) -> None:
        reward_stats_dict = {"mean": reward_mean, "std": reward_std}

        with open(f"{save_loc}/reward_stats.json", "w") as f:
            json.dump(reward_stats_dict, f, indent=4)

    def _eval_reward(self, save_loc: str, n_inits: int = 10) -> None:
        # get sample rewards
        episode_rewards_traj = get_sample_trajectories(
            self.eval_env,
            self.model,
            n_samples=n_inits,
            verbose=True,
        )[2]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(episode_rewards_traj)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Reward")

        fig.tight_layout()
        plt.savefig(f"{save_loc}/all_reward_traj.png")
        plt.close()

        episode_rewards: np.ndarray = episode_rewards_traj.sum(axis=0)

        # save data to json
        if self.verbose:
            print_line("Saving data")

        save_data = {
            "mean": episode_rewards.mean(),
            "sd": episode_rewards.std(),
            "ci": get_ci(episode_rewards),
            "iqm": stats.trim_mean(episode_rewards, 0.25),
            "lqm": stats.trim1(episode_rewards, 0.75, tail="right").mean(),
            "uqm": stats.trim1(episode_rewards, 0.75, tail="left").mean(),
            "l10m": stats.trim1(episode_rewards, 0.90, tail="right").mean(),
            "u10m": stats.trim1(episode_rewards, 0.90, tail="left").mean(),
            "min": episode_rewards.min(),
            "max": episode_rewards.max(),
            "raw": episode_rewards.tolist(),
        }

        with open(f"{save_loc}/reward_stats.json", "w") as f:
            json.dump(save_data, f, indent=4)

        # get reward distribution
        if self.verbose:
            print_line("Getting reward distribution")

        tau_vals = np.arange(
            max(math.floor(episode_rewards.min() * 0.9), 0),
            min(math.ceil(episode_rewards.max() * 1.1), episode_rewards_traj.shape[0]),
            0.1,
        )
        score_distributions, score_distributions_cis = rly.create_performance_profile(
            {type(self.model).__name__.lower(): episode_rewards[:, None]},
            tau_vals,
        )

        # plot results
        if self.verbose:
            print_line("Plotting results")
        fig, ax = plt.subplots(figsize=(12, 12))
        plot_utils.plot_performance_profiles(
            score_distributions,
            tau_vals,
            performance_profile_cis=score_distributions_cis,
            ax=ax,
        )
        ax.set_title(" ".join("_".join(self.env_name).split("_")[:2]))
        ax.legend()
        fig.tight_layout()
        plt.savefig(f"{save_loc}/reward_performance_profile.png")
        plt.close()

        aggregate_func = lambda x: np.array(
            [
                metrics.aggregate_median(x),
                metrics.aggregate_iqm(x),
                metrics.aggregate_mean(x),
            ]
        )

        aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
            {type(self.model).__name__.lower(): episode_rewards[:, None]},
            aggregate_func,
            reps=50000,
        )

        fig, axes = plot_utils.plot_interval_estimates(
            aggregate_scores,
            aggregate_score_cis,
            metric_names=["Median", "IQM", "Mean"],
            algorithms=[type(self.model).__name__.lower()],
            xlabel="Human Normalized Score",
            row_height=8,
            subfigure_width=5,
        )

        plt.subplots_adjust(left=0.1)
        plt.savefig(f"{save_loc}/reward_metrics.png")
        plt.close()

        if self.verbose:
            clear_line()

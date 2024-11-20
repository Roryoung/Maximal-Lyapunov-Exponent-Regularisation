import os
import json

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from src.evaluation.base_eval import Base_Collate


class Plt_MLE_Spectrum_Collate(Base_Collate):
    all_metrics = [
        "PLE",
        "Lyapunov Time (base e, seconds)",
        "Lyapunov Time (base 10, seconds)",
        "Lyapunov Time (base 10, steps)",
        "MLE",
        "# Positive Exponents",
        "SLE",
        "Lyapunov Dimension",
        "Holder Exponent",
    ]

    # all_metrics = ["reward_MLE"]

    def _get_eval_function_results(self, save_loc: str):
        save_path = f"{save_loc}/mle_spectrum/lyapunov_metrics.json"
        # save_path = f"{save_loc}/mle_spectrum/reward_lyapunov_metrics.json"
        if not os.path.isfile(save_path):
            avg_metric_data = None
        else:
            with open(save_path, "r") as results_file:
                avg_metric_data = json.load(results_file)

        return avg_metric_data

    def _collate_results(self, all_results):
        for metric in self.all_metrics:
            self._collate_metric_results(all_results, metric)

    def _collate_metric_results(self, all_results, metric):
        n_envs = len(self.env_list)
        fig, axs = plt.subplots(1, n_envs, figsize=(8, 2.5), sharey=True)
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
                all_reward_values = []
                agent_name = agent_name.replace("_", " ")
                for seed_data in agent_values:
                    for ckpt_name, ckpt_data in seed_data.items():
                        if ckpt_data is None:
                            continue
                        all_metrics = ckpt_data["metrics"]

                        if metric in all_metrics:
                            all_reward_values.append(all_metrics[metric]["mean"])

                all_reward_values = np.array(all_reward_values)

                if all_reward_values.shape[0] == 0:
                    continue

                if agent_name not in registered_agent_names:
                    registered_agent_names.append(agent_name)

                reward_mean = all_reward_values.mean(axis=0)
                lower_ci, upper_ci = stats.t.interval(
                    confidence=0.95,
                    df=all_reward_values.shape[0] - 1,
                    loc=reward_mean,
                    scale=stats.sem(all_reward_values, axis=0),
                )

                print(f"{metric} - {env_name} - {agent_name} - {reward_mean}")
                x_vals = [agent_i - 0.4, agent_i + 0.4]
                x_ticks.append(agent_i)
                axs[env_i].hlines(reward_mean, x_vals[0], x_vals[1], color="black")
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
            axs[env_i].axhline(0, color="tab:red", ls="--")

        axs[0].set_ylabel(metric)
        fig.tight_layout()
        save_dir = f"{self.results_dir}/collated_results/final/mle_spectrum"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{metric}.eps", format="eps")
        plt.close()

    def _collate_series_results(self, all_results):
        raise NotImplementedError()

import os
from typing import Union, Callable
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch as th
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from src.common.utils import print_line, clear_line, append_to_json
from src.evaluation.base_eval import Base_Eval, Base_Collate
from src.models import Dreamer_V3
from src.models.dreamer_v3.distributions import TwoHotEncodingDistribution


def _get_gaussian_perturbation(obs: np.ndarray, model: BaseAlgorithm, eps: float):
    return np.random.normal(0, eps, size=obs.shape)


def _get_fgsm_perturbation(obs: np.ndarray, model: BaseAlgorithm, eps: float):
    model.policy.set_training_mode(False)
    observation, vectorized_env = model.policy.obs_to_tensor(obs)

    observation.requires_grad = True
    if isinstance(model, Dreamer_V3):
        embedded_obs = model.encoder(observation.float())[None, ...]
        updated_prior = model.rssm._representation(
            model.policy.recurrent_state,
            embedded_obs,
            deterministic=True,
        )[1]
        updated_prior = updated_prior.reshape(*updated_prior.shape[:2], -1)

        latent_embedding = th.cat((updated_prior, model.policy.recurrent_state), -1)
        critic_bins = model.critic(latent_embedding)
        values = TwoHotEncodingDistribution(critic_bins, dims=1).mean[0]
    elif isinstance(model, OffPolicyAlgorithm):
        action = _get_fgsm_action(observation, vectorized_env, model)
        values = model.critic(observation, action)[0]
    elif isinstance(model, OnPolicyAlgorithm):
        values = model.policy.predict_values(observation)
    else:
        raise ValueError("Unknown agent")

    avg_value = th.mean(values)

    model.policy.zero_grad()
    avg_value.backward()

    obs_grad = observation.grad.data
    sign_obs_grad = obs_grad.sign()
    return eps * -sign_obs_grad.cpu().numpy()


def _get_fgsm_action(
    observation: th.Tensor,
    vectorized_env: bool,
    model: BaseAlgorithm,
):
    actions = model.policy._predict(observation, deterministic=True)
    actions = actions.reshape((-1, *model.policy.action_space.shape))

    if isinstance(model.policy.action_space, spaces.Box):
        if model.policy.squash_output:
            # Rescale to proper domain when using squashing
            low, high = (
                th.from_numpy(model.policy.action_space.low).to(actions.device),
                th.from_numpy(model.policy.action_space.high).to(actions.device),
            )
            actions = low + (0.5 * (actions + 1.0) * (high - low))
        else:
            # Actions could be on arbitrary scale, so clip the actions to avoid
            # out of bound error (e.g. if sampling from a Gaussian distribution)
            actions = th.clip(
                actions,
                th.from_numpy(model.policy.action_space.low).to(actions.device),
                th.from_numpy(model.policy.action_space.high).to(actions.device),
            )

    # Remove batch dimension if needed
    if not vectorized_env:
        actions = actions.squeeze(axis=0)

    return actions


class Observation_Attack_Collate(Base_Collate):
    def _get_eval_function_results(self, save_loc: str):
        save_path = f"{save_loc}/obs_attack/reward_stats.json"
        if not os.path.isfile(save_path):
            return None

        with open(save_path, "r") as results_file:
            return json.load(results_file)

    def _collate_results(self, all_results):
        self._collate_parameter(
            all_results,
            attack_name="gaussian_noise",
            reuse_perturb=False,
            frame_skip=0,
        )

    def _collate_parameter(
        self,
        all_results,
        attack_name: str,
        reuse_perturb: bool,
        frame_skip: Union[float, None] = None,
        value_threshold: Union[float, None] = None,
    ):
        if frame_skip is None and value_threshold is None:
            raise ValueError

        if frame_skip is not None and value_threshold is not None:
            raise ValueError

        n_envs = len(self.env_list)
        n_cols = 6
        n_rows = n_envs // n_cols + (n_envs % n_cols != 0)
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(2 * n_cols, 5 * n_rows), sharey=True
        )
        cmap = plt.get_cmap("tab10")
        if n_envs <= n_cols:
            axs = np.array([axs])

        registered_agent_names = []

        for i, (env_name, env_values) in enumerate(all_results.items()):
            row_i, col_i = divmod(i, n_cols)
            axs[row_i, col_i].set_title(
                "".join([s[0].capitalize() for s in env_name.split("_")])
            )
            for agent_name, agent_values in env_values.items():
                all_reward_values = []
                all_std_values = []
                for seed_data in agent_values:
                    seed_reward_values = []
                    for ckpt_name, ckpt_data in seed_data.items():
                        if ckpt_data is None:
                            continue

                        for attack, attack_values in ckpt_data.items():
                            is_attack = attack_name in attack
                            is_frame_skip = (
                                frame_skip is not None
                                and f"skip={frame_skip}," in attack
                            )
                            is_value_threshold = (
                                value_threshold is not None
                                and f"value_threshold={value_threshold}," in attack
                            )
                            is_reuse_perturb = f"reuse={reuse_perturb}" in attack
                            if (
                                is_attack
                                and (is_frame_skip or is_value_threshold)
                                and is_reuse_perturb
                            ):
                                all_reward_values.append(attack_values["reward_values"])
                                all_std_values.append(attack_values["noise_std_values"])

                all_reward_values = np.concatenate(all_reward_values, axis=-1).T
                all_std_values = np.array(all_std_values)

                if all_std_values.shape[0] == 0 or all_reward_values.shape[0] == 0:
                    continue

                # check for differing std values
                assert np.all(all_std_values[0] == all_std_values)

                if agent_name not in registered_agent_names:
                    registered_agent_names.append(agent_name)

                reward_mean = all_reward_values.mean(axis=0)
                lower_ci, upper_ci = stats.t.interval(
                    confidence=0.95,
                    df=all_reward_values.shape[0] - 1,
                    loc=reward_mean,
                    scale=stats.sem(all_reward_values, axis=0),
                )

                axs[row_i, col_i].plot(
                    all_std_values[0],
                    reward_mean,
                    label=agent_name if i == 0 else "",
                    color=cmap(registered_agent_names.index(agent_name)),
                )
                axs[row_i, col_i].fill_between(
                    all_std_values[0],
                    lower_ci,
                    upper_ci,
                    alpha=0.3,
                    color=cmap(registered_agent_names.index(agent_name)),
                )

            axs[row_i, col_i].set_ylim(0, 1000)
            axs[row_i, col_i].set_xlabel(r"$\sigma$")
            axs[row_i, 0].set_ylabel("Total Reward")

        title = f"Observation attack using {attack_name}"
        if value_threshold is not None:
            title += f" (Value Threshold = {value_threshold}, reuse perturb = {reuse_perturb})"
        else:
            title += f" (Frame Skip = {frame_skip}, reuse perturb = {reuse_perturb})"

        fig.tight_layout()

        # add legend
        fig.subplots_adjust(bottom=0.25)
        axs[0, 0].legend(
            loc="upper center",
            bbox_to_anchor=(3.5, -0.2),
            fancybox=False,
            shadow=False,
            ncol=4,
        )

        # save results
        save_dir = f"{self.results_dir}/collated_results/final/obs_attack"
        save_dir += f"/{attack_name}/reuse_perturb={reuse_perturb}"
        if value_threshold is not None:
            save_dir = f"{save_dir}/value_based"
            save_file_name = f"value_threshold={value_threshold}"
        else:
            save_dir = f"{save_dir}/frame_skip"
            save_file_name = f"skip={frame_skip}"

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{save_file_name}.pdf", format="pdf")
        plt.close()

    def _collate_series_results(self, all_results):
        raise NotImplementedError


class Observation_Attack(Base_Eval):
    all_frame_skips = [0]
    value_fn_threshold_values = [0.5, 1, 1.5, 2]
    reuse_perturb_values = [
        False,
        # True,
    ]
    all_attack_methods = [
        # _get_fgsm_perturbation,
        _get_gaussian_perturbation,
    ]
    all_attack_names = [
        # "fgsm",
        "gaussian_noise",
    ]
    env_max_eps = {
        ("point_mass", "easy"): 0.1,
        ("cartpole", "balance"): 0.4,
        ("cartpole", "swingup"): 0.4,
        ("walker", "stand"): 1,
        ("walker", "walk"): 1,
        ("walker", "run"): 1,
        ("cheetah", "run"): 0.1,
    }
    resolution = 10

    @classmethod
    def get_collate_results_class(self) -> Union[Base_Collate, None]:
        return Observation_Attack_Collate

    @classmethod
    def get_eval_name(self):
        return "obs_attack"

    def _plt_all_figs(self, save_loc: str) -> None:
        pass

    def _save_metrics(self, save_loc: str) -> None:
        for attack_method, attack_name in zip(
            self.all_attack_methods, self.all_attack_names
        ):
            for reuse_perturb in self.reuse_perturb_values:
                # run frame skip tests
                for frame_skip in self.all_frame_skips:
                    self._run_attack(
                        save_loc,
                        n_inits=80,
                        attack_method=attack_method,
                        attack_name=attack_name,
                        frame_skip=frame_skip,
                        reuse_perturbation=reuse_perturb,
                    )

                # # run threshold value tests
                # if isinstance(self.model, OffPolicyAlgorithm):
                #     for value_threshold in self.value_fn_threshold_values:
                #         self._run_attack(
                #             save_loc,
                #             n_inits=8,
                #             attack_method=attack_method,
                #             attack_name=attack_name,
                #             value_threshold=value_threshold,
                #             reuse_perturbation=reuse_perturb,
                #         )

    def _run_attack(
        self,
        save_loc: str,
        attack_method: Callable[[np.ndarray, BaseAlgorithm], np.ndarray],
        attack_name: str,
        n_inits: int = 8,
        frame_skip: Union[int, None] = None,
        value_threshold: Union[float, None] = None,
        reuse_perturbation: bool = False,
    ):
        if value_threshold is None and frame_skip is None:
            raise ValueError("`value_threshold` and `frame_skip` cannot both be `None`")

        if value_threshold is not None and frame_skip is not None:
            raise ValueError(
                "`value_threshold` and `frame_skip` cannot both be not `None`"
            )

        max_eps = self.env_max_eps[self.env_name]
        eps_values = np.arange(
            0,
            max_eps + max_eps / (self.resolution + 1),
            max_eps / self.resolution,
        )

        _, reward_trajs, episode_n_perturbations = self.get_attacked_trajs(
            attack_method,
            attack_name,
            n_inits,
            frame_skip,
            value_threshold,
            reuse_perturbation,
            eps_values,
        )

        if self.verbose:
            clear_line()

        # plot results
        self.plot_results(
            save_loc,
            attack_name,
            frame_skip,
            value_threshold,
            reuse_perturbation,
            eps_values,
            reward_trajs,
            episode_n_perturbations,
        )

        # save values
        action_noise_dict = {
            "noise_std_values": eps_values.tolist(),
            "reward_means": reward_trajs.mean(axis=1).tolist(),
            "reward_std": reward_trajs.std(axis=1).tolist(),
            "reward_values": reward_trajs.tolist(),
        }
        key = f"obs_{attack_name}"
        if value_threshold is not None:
            key += f"(value_threshold={value_threshold}, reuse={reuse_perturbation})"
        else:
            key += f"(skip={frame_skip}, reuse={reuse_perturbation})"

        append_to_json(
            f"{save_loc}/reward_stats.json",
            key,
            action_noise_dict,
        )

    def get_attacked_trajs(
        self,
        attack_method,
        attack_name,
        n_inits,
        frame_skip,
        value_threshold,
        reuse_perturbation,
        eps_values,
    ):
        n_envs = self.eval_env.num_envs
        n_inits = (n_inits // n_envs) + (n_inits % n_envs != 0)

        all_episode_rewards, all_episode_n_perturbations = [], []
        all_state_traj = []
        for eps in eps_values:
            episode_rewards, episode_n_perturbations = [], []
            episode_state_traj = []
            for init_i in range(n_inits):
                print_stmt = f"{attack_name} attack, "
                if value_threshold is not None:
                    print_stmt += f"value_threshold={value_threshold}, "
                else:
                    print_stmt += f"frame_skip={frame_skip}, "

                print_stmt += f"reuse_perturb={reuse_perturbation}, "
                print_stmt += f"Perturb eps={eps}"
                print_stmt += f"[{init_i}/{n_inits}]"
                print_line(print_stmt)

                current_rewards = np.zeros(n_envs)
                current_n_perturbations = np.zeros(n_envs)
                done = np.full(n_envs, False)
                obs = self.eval_env.reset()
                if isinstance(self.model, Dreamer_V3):
                    self.model.policy.init_states(np.full(n_envs, True))

                perturbation = np.zeros_like(obs)
                state_traj = []
                t = 0

                while not np.any(done):
                    state_traj.append(np.array(self.eval_env.env_method("get_state")))

                    perturbation = self.get_perturbation(
                        perturbation=perturbation,
                        eps=eps,
                        attack_method=attack_method,
                        frame_skip=frame_skip,
                        value_threshold=value_threshold,
                        reuse_perturbation=reuse_perturbation,
                        obs=obs,
                        t=t,
                    )

                    current_n_perturbations += 1 * np.all(
                        perturbation != np.zeros_like(perturbation), axis=1
                    )
                    obs += perturbation
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    current_rewards += reward
                    t += 1

                episode_rewards.append(current_rewards)
                episode_n_perturbations.append(current_n_perturbations)
                episode_state_traj.append(state_traj)
            all_episode_rewards.append(np.concatenate(episode_rewards))
            all_episode_n_perturbations.append(np.concatenate(episode_n_perturbations))
            all_state_traj.append(np.concatenate(episode_state_traj, axis=1))
        all_episode_rewards = np.array(all_episode_rewards)
        all_episode_n_perturbations = np.array(all_episode_n_perturbations)
        all_state_traj = np.array(all_state_traj)
        return all_state_traj, all_episode_rewards, all_episode_n_perturbations

    def plot_results(
        self,
        save_loc: str,
        attack_name: str,
        frame_skip: Union[int, None],
        value_threshold: Union[float, None],
        reuse_perturbation: bool,
        eps_values: np.ndarray,
        all_episode_rewards: np.ndarray,
        all_episode_n_perturbations: np.ndarray,
    ):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        mean = all_episode_rewards.mean(axis=1)
        lower_ci, upper_ci = stats.t.interval(
            confidence=0.95,
            df=all_episode_rewards.shape[1] - 1,
            loc=mean,
            scale=stats.sem(all_episode_rewards, axis=1),
        )

        ax.plot(
            eps_values,
            mean,
            label=f"{self.model.__class__.__name__}: Reward",
        )
        ax.fill_between(
            eps_values,
            lower_ci,
            upper_ci,
            alpha=0.5,
        )

        # add axis labels
        ax.set_xlabel("Perturbation eps")
        ax.set_ylabel("Episode Reward")
        title = f"Obs attack using {attack_name}"
        if value_threshold is not None:
            title += f" (Value Threshold = {value_threshold}, reuse perturb = {reuse_perturbation})"
        else:
            title += (
                f" (Frame Skip = {frame_skip}, reuse perturb = {reuse_perturbation})"
            )

        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        save_folder = f"{save_loc}/{attack_name}/reuse_perturb={reuse_perturbation}"
        if value_threshold is not None:
            save_folder = f"{save_folder}/value_based"
            save_file_name = f"value_threshold={value_threshold}.png"
        else:
            save_folder = f"{save_folder}/frame_skip"
            save_file_name = f"skip={frame_skip}.png"

        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f"{save_folder}/{save_file_name}")
        plt.close()

    def get_perturbation(
        self,
        perturbation: np.ndarray,
        eps: float,
        attack_method: Callable[[np.ndarray, BaseAlgorithm], np.ndarray],
        frame_skip: Union[int, None],
        value_threshold: Union[float, None],
        reuse_perturbation: bool,
        obs: np.ndarray,
        t: int,
    ):
        if not reuse_perturbation:
            perturbation = np.zeros_like(obs)

        if frame_skip is not None and t % (frame_skip + 1) == 0:
            perturbation = attack_method(obs, self.model, eps)
        elif value_threshold is not None:
            action = self.model.predict(obs, deterministic=True)[0]
            random_action = np.stack(
                [self.eval_env.action_space.sample() for _ in range(obs.shape[0])]
            )

            true_value = self.model.critic(
                th.tensor(obs).to(self.model.device),
                th.from_numpy(action).to(self.model.device),
            )
            random_value = self.model.critic(
                th.tensor(obs).to(self.model.device),
                th.from_numpy(random_action).float().to(self.model.device),
            )
            significant_value = (true_value[0] - random_value[0]) > value_threshold
            significant_value = significant_value.cpu().numpy()
            obs_perturbation = significant_value * attack_method(obs, self.model)

            if reuse_perturbation:
                obs_perturbation += np.invert(significant_value) * perturbation
            perturbation = obs_perturbation

        return perturbation

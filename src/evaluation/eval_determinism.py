import os
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from src.evaluation.utils import get_sample_trajectories
from src.evaluation.base_eval import Base_Eval, Base_Collate


class Eval_Determinism(Base_Eval):
    @classmethod
    def get_collate_results_class(cls) -> Union[Base_Collate, None]:
        return None

    @classmethod
    def get_eval_name(self):
        return "determinism"

    def _plt_all_figs(self, save_loc: str) -> None:
        pass

    def _save_metrics(self, save_loc: str) -> None:
        self._plot_state_trajs(save_loc, n_inits=3)
        self._plot_reward_trajs(save_loc)

    def _plot_state_trajs(self, save_loc: str, n_inits: int = 8):
        """Plot the state trajectories subject to an initial state perturbation"""

        # get original state
        self.eval_env.reset()
        init_state: np.ndarray = self.eval_env.env_method("get_state")[0]
        init_state = init_state[np.newaxis, :].repeat(n_inits, axis=0)

        # add perturbation
        perturb_points = np.random.randn(*init_state.shape)
        perturb_points /= np.linalg.norm(perturb_points, axis=1, keepdims=True)
        init_state += 1e-4 * perturb_points

        # get state trajectories
        state_traj = get_sample_trajectories(
            self.eval_env,
            self.model,
            initial_states=init_state,
            max_len=101,
        )[0]

        n_dims = state_traj.shape[-1]
        n_dims = min(5, n_dims)
        fig, axs = plt.subplots(n_dims, figsize=(4, 3), sharex=True)

        for dim_i in range(n_dims):
            axs[dim_i].plot(state_traj[:, :, dim_i])
            axs[dim_i].set_ylabel(dim_i)
        axs[-1].set_xlabel("Timestep")

        fig.tight_layout()
        fig.align_labels(axs)
        # plt.savefig(f"{save_loc}/state_traj.png")
        plt.savefig(f"{save_loc}/state_traj.eps", format="eps")
        plt.close()

    def _plot_reward_trajs(self, save_loc: str, n_inits: int = 10_000):
        """Plot the reward trajectories subject to an initial state perturbation"""

        # get original state
        self.eval_env.reset()
        init_state: np.ndarray = self.eval_env.env_method("get_state")[0]
        init_state = init_state[np.newaxis, :].repeat(n_inits, axis=0)

        # add perturbation
        perturbation /= np.linalg.norm(perturbation, axis=0, keepdims=True)
        perturbation = perturbation[np.newaxis, :].repeat(n_inits, axis=0)
        eps_values = np.linspace(0, 0.001, n_inits)[:, np.newaxis]
        init_state += eps_values * perturbation

        if os.path.isfile(f"{save_loc}/reward_data.npy"):
            reward_traj = np.load(f"{save_loc}/reward_data.npy")
        else:
            # get state and reward trajectories
            reward_traj = get_sample_trajectories(
                self.eval_env,
                self.model,
                initial_states=init_state,
                verbose=True,
                max_len=501,
            )[2]
            np.save(f"{save_loc}/reward_data.npy", reward_traj)

        fig, ax = plt.subplots(1, 2, figsize=(8, 3))

        # plot return fractal
        total_rewards = reward_traj.sum(axis=0)
        ax[0].scatter(eps_values[:, 0], total_rewards, s=1)
        ax[0].set_xlabel("Perturbation Size")
        ax[0].set_ylabel("Total Reward")
        ax[0].set_xticks([0, 0.0005, 0.001])

        # plot best and worst reward trajectories
        reward_traj_inds = []
        reward_traj_inds += np.argsort(total_rewards)[-3:].tolist()
        reward_traj_inds += np.argsort(total_rewards)[:3].tolist()
        ax[1].plot(reward_traj[:, np.array(reward_traj_inds)])
        ax[1].set_xlabel("Steps")
        ax[1].set_ylabel("Reward")

        # save fig
        fig.tight_layout()
        plt.savefig(f"{save_loc}/all_reward_traj.png")
        plt.savefig(f"{save_loc}/all_reward_traj.eps", format="eps")
        plt.close()

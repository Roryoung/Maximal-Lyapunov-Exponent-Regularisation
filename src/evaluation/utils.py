from typing import Union, Tuple, List, Optional

import numpy as np
from scipy.optimize import minimize
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.base_class import BaseAlgorithm

from src.common.utils import print_line, clear_line
from src.models import Dreamer_V3


def get_sample_trajectories(
    env: Union[VecEnv, VecNormalize],
    agent: BaseAlgorithm,
    initial_states: Union[np.ndarray, None] = None,
    n_samples: Union[int, None] = None,
    return_img: bool = False,
    verbose: bool = False,
    max_len: Union[int, None] = None,
    init_n_steps: Optional[int] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    if n_samples is None and initial_states is None:
        raise RuntimeError(
            "`n_samples` and `initial_states` are both `None` when one must be specified."
        )

    n_samples = initial_states.shape[0] if initial_states is not None else n_samples
    n_envs = env.num_envs
    n_inits = (n_samples // n_envs) + (n_samples % n_envs != 0)
    init_n_steps = init_n_steps or np.inf

    all_state_list, all_obs_list, all_action_list = [], [], []
    all_reward_list, all_img_list = [], []

    for init_i in range(n_inits):
        if verbose:
            print_line(f"Sampling states: [{init_i}/{n_inits}]")

        init_state_list, init_obs_list, init_action_list = [], [], []
        init_reward_list, init_img_list = [], []

        obs = env.reset()
        done = False
        if isinstance(agent, (Dreamer_V3)):
            agent.policy.init_states(np.full(env.num_envs, True))

        if initial_states is not None:
            new_obs = []
            for env_i in range(n_envs):
                state_index = init_i * n_envs + env_i
                if state_index >= n_samples:
                    new_obs.append([obs[env_i]])
                else:
                    po = env.env_method(
                        "set_state", state=initial_states[state_index], indices=env_i
                    )
                    if isinstance(env, VecNormalize):
                        po = env.normalize_obs(po)
                    new_obs.append(po)
            obs = np.concatenate(new_obs)

        # produce trajectories
        n_steps = 0
        while True:
            if verbose:
                print_line(
                    f"Sampling states: [{init_i}/{n_inits}][{n_steps}/{max_len or 1000}]"
                )

            init_state_list.append(np.stack(env.env_method("get_state")))
            init_obs_list.append(obs)
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            init_reward_list.append(reward)
            init_action_list.append(action)

            if isinstance(agent, (Dreamer_V3)):
                if np.any(done):
                    agent.policy.init_states(done)

            n_steps += 1

            if return_img:
                init_img_list.append(np.array(env.get_images()))

            if np.any(done):
                if max_len is None:
                    break

            if max_len is not None:
                if n_steps >= max_len:
                    break

                env_state = env.env_method("get_state")
                env.reset()

                obs = []
                for env_i, state in enumerate(env_state):
                    o = env.env_method("set_state", indices=env_i, state=state)
                    if isinstance(env, VecNormalize):
                        o = env.normalize_obs(o)
                    obs.append(o)
                obs = np.concatenate(obs)

            if n_steps % init_n_steps == 0:
                if isinstance(agent, (Dreamer_V3)):
                    print("resetting")
                    agent.policy.init_states(np.full(env.num_envs, True))

        # collate trajectories results
        all_state_list.append(np.array(init_state_list))
        all_obs_list.append(np.array(init_obs_list))
        all_reward_list.append(np.array(init_reward_list))
        all_action_list.append(np.array(init_action_list))
        if return_img:
            all_img_list.append(np.array(init_img_list))

    all_state_list = np.concatenate(all_state_list, axis=1)[:, :n_samples]
    all_obs_list = np.concatenate(all_obs_list, axis=1)[:, :n_samples]
    all_reward_list = np.concatenate(all_reward_list, axis=1)[:, :n_samples]
    all_action_list = np.concatenate(all_action_list, axis=1)[:, :n_samples]
    if return_img:
        all_img_list = np.concatenate(all_img_list, axis=1)[:, :n_samples]

    if verbose:
        clear_line()
    if return_img:
        return (
            all_state_list,
            all_obs_list,
            all_reward_list,
            all_action_list,
            all_img_list,
        )

    return all_state_list, all_obs_list, all_reward_list, all_action_list


def get_always_increasing_dimensions_mask(
    trajs: np.ndarray,
    step_length: int,
    mask_threshold: float = 0.95,
    start_point: int = 1,
    decreasing: bool = False,
) -> List[bool]:
    step_diff_trajs = trajs[::step_length]
    step_diff = step_diff_trajs[start_point:] - step_diff_trajs[:-start_point]

    if decreasing:
        step_diff = -step_diff

    step_increasing = np.mean(step_diff > 0, axis=0) > mask_threshold
    all_step_increasing = np.mean(step_increasing, axis=0) > mask_threshold

    return all_step_increasing


def get_monotonic_dimensions_mask(
    trajs,
    step_length: int,
    mask_threshold: float = 0.95,
    start_point: int = 1,
    return_invert: bool = False,
) -> List[bool]:
    increasing_dims = get_always_increasing_dimensions_mask(
        trajs,
        step_length,
        mask_threshold,
        start_point,
        decreasing=False,
    )

    decreasing_dims = get_always_increasing_dimensions_mask(
        trajs,
        step_length,
        mask_threshold,
        start_point,
        decreasing=True,
    )

    if return_invert:
        return np.invert(increasing_dims + decreasing_dims)

    return increasing_dims + decreasing_dims


def mask_monotonic_dimensions(
    trajs,
    step_length: int,
    mask_threshold: float = 0.95,
    start_point: int = 1,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    axis_mask = get_monotonic_dimensions_mask(
        trajs,
        step_length=step_length,
        mask_threshold=mask_threshold,
        start_point=start_point,
        return_invert=True,
    )

    masked_dimensions = np.arange(len(axis_mask))[np.invert(axis_mask)]
    if verbose:
        print(f"Removing dimensions {masked_dimensions}")

    return trajs[:, :, axis_mask], masked_dimensions


def get_trajectory_divergence(traj, true_traj):
    """Return the euclidean distance between two trajectories"""
    return np.linalg.norm(traj - true_traj, axis=-1).mean(axis=1)


def mle_model(params, input_vals):
    """Return the clipped mle function"""
    slope, intercept, maximum_val = params
    return np.minimum(slope * input_vals + intercept, maximum_val)


def objective_function(params, input_vals, true_vals):
    """Return the weighted MSE error for a given set of parameters"""
    predicted_vals = mle_model(params, input_vals)
    time_reg = np.linspace(1, 0, predicted_vals.shape[0] + 1)[:-1]
    residuals = (predicted_vals - true_vals) * time_reg
    return 0.5 * np.mean(residuals**2)


def get_mle_params(initial_params, x_vals, y_vals):
    """Optimize the MLE objective function and return the parameters"""
    return minimize(objective_function, initial_params, args=(x_vals, y_vals)).x

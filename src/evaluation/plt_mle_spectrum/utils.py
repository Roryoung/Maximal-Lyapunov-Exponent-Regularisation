import os
from typing import Union, Tuple, Dict
import math

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from src.common.utils import clear_line, print_line
from src.evaluation.utils import get_sample_trajectories
from src.models import Dreamer_V3


def get_sample_initial_state(
    env: VecNormalize, agent, n_samples, low_time: int, upper_time: int
):
    sample_states = get_sample_trajectories(
        env,
        agent,
        n_samples=n_samples,
    )[0]
    start_t_percent = np.random.uniform(low_time, upper_time)
    initial_state_loc = int(start_t_percent * sample_states.shape[0])
    return sample_states[initial_state_loc, :, :]


def get_sample_estimated_lyapunov_exponents(
    env: VecNormalize,
    agent,
    n_iterations: int,
    norm_period: int,
    eps: float,
    dt: float,
    lower_time: int,
    upper_time: int,
    n_lyap_exps: Union[int, None] = None,
    verbose: bool = False,
    n_samples: Union[int, None] = None,
    initial_states: Union[np.ndarray, None] = None,
):
    if n_samples is None and initial_states is None:
        raise RuntimeError(
            "`n_samples` and `initial_states` are both `None` when one must be specified."
        )

    n_envs = env.num_envs
    if initial_states is None:
        initial_states = get_sample_initial_state(
            env,
            agent,
            n_samples,
            lower_time,
            upper_time,
        )
    n_samples = initial_states.shape[0]
    n_repeats = int(math.ceil(n_samples / n_envs))
    max_samples = n_repeats * n_envs
    n_lyap_exps = n_lyap_exps or env.env_method("get_state")[0].shape[-1]

    all_lyap_exps = np.empty((max_samples, n_lyap_exps))
    all_lyap_exps_hist = np.empty((n_iterations, max_samples, n_lyap_exps))

    all_reward_lyap_exps = np.empty((max_samples, n_lyap_exps))
    all_reward_lyap_exps_hist = np.empty((n_iterations, max_samples, n_lyap_exps))

    # for sample_no in range(n_samples):
    for sample_no in range(n_repeats):
        if verbose:
            print_line(f"Estimating exponents [{sample_no}]/[{n_repeats}]")
        # get initial state samples

        for env_i in range(n_envs):
            state_index = min(sample_no * n_envs + env_i, n_samples - 1)
            env.env_method(
                "set_state",
                state=initial_states[state_index],
                indices=env_i,
            )

        # get batched estimated lyapunov exponents
        (
            lyap_exps,
            lyap_exps_history,
            reward_lyap_exps,
            reward_lyap_exps_history,
        ) = batch_estimate_lyapunov_exponents(
            env=env,
            agent=agent,
            n_iterations=n_iterations,
            norm_period=norm_period,
            eps=eps,
            dt=dt,
            n_lyap_exps=n_lyap_exps,
            verbose=verbose,
        )
        sample_lower = sample_no * n_envs
        sample_upper = (sample_no + 1) * n_envs
        all_lyap_exps[sample_lower:sample_upper] = lyap_exps
        all_lyap_exps_hist[:, sample_lower:sample_upper, :] = lyap_exps_history

        all_reward_lyap_exps[sample_lower:sample_upper] = reward_lyap_exps
        all_reward_lyap_exps_hist[:, sample_lower:sample_upper, :] = (
            reward_lyap_exps_history
        )

    if verbose:
        clear_line()

    all_lyap_exps = all_lyap_exps[:n_samples]
    all_lyap_exps_hist = all_lyap_exps_hist[:, :n_samples]

    all_reward_lyap_exps = all_reward_lyap_exps[:n_samples]
    all_reward_lyap_exps_hist = all_reward_lyap_exps_hist[:, :n_samples]

    return (
        all_lyap_exps,
        all_lyap_exps_hist,
        all_reward_lyap_exps,
        all_reward_lyap_exps_hist,
    )


def batch_estimate_lyapunov_exponents(
    env: Union[VecEnv, VecNormalize],
    agent,
    n_iterations: int,
    norm_period: int,
    eps: float,
    dt: float,
    n_lyap_exps: Union[int, None] = None,
    stop_early: bool = True,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    # get original state
    env.reset()
    original_state = np.array(env.env_method("get_state"))
    n_samples, n_dims = original_state.shape
    if isinstance(agent, Dreamer_V3):
        agent.policy.init_states(np.full(env.num_envs, True))
        true_dreamer_actions = agent.policy.actions.clone().detach()
        true_dreamer_recurrent = agent.policy.recurrent_state.clone().detach()
        true_dreamer_stochastic = agent.policy.stochastic_state.clone().detach()

    # get perturbed states
    n_lyap_exps = n_lyap_exps or n_dims
    estimated_lyap_exps = np.empty((n_iterations, n_lyap_exps, n_samples))
    estimated_reward_lyap_exps = np.empty((n_iterations, n_lyap_exps, n_samples))
    perturbed_states = []
    for _ in range(n_lyap_exps):
        perturbation = np.random.rand(n_samples, n_dims)
        perturbation /= np.linalg.norm(perturbation, axis=1, keepdims=True)
        perturbed_states.append(original_state + eps * perturbation)

    total_log_norm_vals = 0
    total_log_reward = 0
    eps = np.double(eps)
    total_reward_samples = 0
    for i in range(n_iterations):
        if verbose:
            print_line(f"[{i}/{n_iterations}]")

        # set current true state
        env.reset()
        obs = [
            env.env_method("set_state", state=state, indices=state_i)
            for state_i, state in enumerate(original_state)
        ]
        obs = np.concatenate(obs)
        obs = env.normalize_obs(obs) if isinstance(env, VecNormalize) else obs

        # set dreamer internal state
        if isinstance(agent, (Dreamer_V3)):
            agent.policy.actions = true_dreamer_actions
            agent.policy.recurrent_state = true_dreamer_recurrent
            agent.policy.stochastic_state = true_dreamer_stochastic

        # get next true step
        for norm_period_i in range(norm_period):
            action = agent.predict(obs, deterministic=True)[0]
            obs, true_reward, *_ = env.step(action)
            if norm_period_i == 0:
                original_reward = true_reward

        # get next true state
        original_state = np.array(env.env_method("get_state"))
        if isinstance(agent, Dreamer_V3):
            next_dreamer_actions = agent.policy.actions.clone().detach()
            next_dreamer_recurrent = agent.policy.recurrent_state.clone().detach()
            next_dreamer_stochastic = agent.policy.stochastic_state.clone().detach()

        # iterate over perturbed states
        updated_perturbed_states = []
        updated_perturbed_rewards = []
        all_original_perturb_rewards = []
        for perturbed_state in perturbed_states:
            # set current perturbed state
            obs = [
                env.env_method("set_state", state=state, indices=state_i)
                for state_i, state in enumerate(perturbed_state)
            ]
            obs = np.concatenate(obs)
            obs = env.normalize_obs(obs) if isinstance(env, VecNormalize) else obs

            if isinstance(agent, (Dreamer_V3)):
                agent.policy.actions = true_dreamer_actions
                agent.policy.recurrent_state = true_dreamer_recurrent
                agent.policy.stochastic_state = true_dreamer_stochastic

            # get perturbed next steps
            for norm_period_i in range(norm_period):
                action = agent.predict(obs, deterministic=True)[0]
                obs, perturb_reward, *_ = env.step(action)

                if norm_period_i == 0:
                    all_original_perturb_rewards.append(perturb_reward)

            updated_perturbed_states.append(np.array(env.env_method("get_state")))
            updated_perturbed_rewards.append(perturb_reward)

        # calculate divergence
        distance = [
            (perturbed_state - original_state)
            for perturbed_state in updated_perturbed_states
        ]

        reward_distance = [
            (perturb_reward - true_reward) / (original_perturb_reward - original_reward)
            for (perturb_reward, original_perturb_reward) in zip(
                updated_perturbed_rewards, all_original_perturb_rewards
            )
        ]
        reward_distance = np.array(reward_distance)[:, :, None]
        log_reward_norm_vals = np.log(np.linalg.norm(reward_distance, axis=-1))

        # orthonormalize
        dist_norm, log_norm_vals = batch_orthonormalize(distance / eps)

        # update perturbed states
        perturbed_states = [original_state + eps * x_i for x_i in dist_norm]

        # update internal representation
        if isinstance(agent, (Dreamer_V3)):
            agent.policy.actions = next_dreamer_actions
            agent.policy.recurrent_state = next_dreamer_recurrent
            agent.policy.stochastic_state = next_dreamer_stochastic

        # record total log state norm
        total_log_norm_vals += log_norm_vals
        estimated_lyap_exps[i] = total_log_norm_vals / ((i + 1) * dt * norm_period)

        # record total log reward norm
        is_numeric = np.invert(
            np.isnan(log_reward_norm_vals)
            + np.isinf(log_reward_norm_vals)
            + np.isneginf(log_reward_norm_vals)
        )
        total_reward_samples += 1 * is_numeric

        total_log_reward += np.nan_to_num(log_reward_norm_vals, posinf=0, neginf=0)
        estimated_reward_lyap_exps[i] = total_log_reward / (
            total_reward_samples * dt * norm_period
        )

    estimated_lyap_exps = estimated_lyap_exps.swapaxes(1, 2)
    final_lyap_exps = estimated_lyap_exps[-1]

    estimated_reward_lyap_exps = estimated_reward_lyap_exps.swapaxes(1, 2)
    final_reward_lyap_exps = estimated_reward_lyap_exps[-1]

    return (
        final_lyap_exps,
        estimated_lyap_exps,
        final_reward_lyap_exps,
        estimated_reward_lyap_exps,
    )


def batch_orthonormalize(relative_dist):
    """Compute an orthogonal basis and the exponential change
    in the norm along each element of the basis."""

    n_lyap_exps = len(relative_dist)
    n_samples, n_dims = relative_dist[0].shape

    updated_basis = np.empty((n_lyap_exps, n_samples, n_dims))
    norm_vals = np.empty((n_lyap_exps, n_samples))

    norm_vals[0] = np.linalg.norm(relative_dist[0], axis=-1)
    updated_basis[0] = relative_dist[0] / norm_vals[0][:, None]

    for i in range(1, n_lyap_exps):
        temp = 0
        for j in range(i):
            temp += (
                np.sum(updated_basis[j] * relative_dist[i], axis=1, keepdims=True)
                * updated_basis[j]
            )
        updated_basis[i] = relative_dist[i] - temp

        # normalize
        norm_vals[i] = np.linalg.norm(updated_basis[i], axis=-1)
        updated_basis[i] /= norm_vals[i][:, None]

    return updated_basis, np.log(norm_vals)


def get_series_statistics(series: np.ndarray, return_tuple: bool = True):
    mean = series.mean(axis=0)
    std = series.std(axis=0)

    lower_ci, upper_ci = stats.t.interval(
        confidence=0.95,
        df=series.shape[0] - 1,
        loc=mean,
        scale=stats.sem(series, axis=0),
    )
    conf_interval = (lower_ci, upper_ci)

    if return_tuple:
        return mean, std, conf_interval

    return {"mean": mean, "std": std, "ci": conf_interval}


def get_sum_of_positive_lyap_exponents(lyap_exponents: np.ndarray) -> np.ndarray:
    return np.clip(lyap_exponents, 0, None).sum(axis=1)


def get_maximum_lyap_exponent(lyap_exponents: np.ndarray) -> np.ndarray:
    return lyap_exponents.max(axis=1)


def get_lyap_time(lyap_exponents: np.ndarray, base=np.e, dt=1) -> np.ndarray:
    mle = get_maximum_lyap_exponent(lyap_exponents)
    lyap_time_steps = np.log(base) / mle
    lyap_time_seconds = lyap_time_steps / dt
    return np.where(lyap_time_seconds > 0, lyap_time_seconds, np.inf)


def get_count_positive_lyap_exponents(
    lyap_exponents: np.ndarray,
    margin: float = 0.0,
) -> np.ndarray:
    return np.sum(lyap_exponents > margin, axis=1)


def get_sum_of_lyapunov_exponents(lyap_exponents: np.ndarray) -> np.ndarray:
    return lyap_exponents.sum(axis=1)


def get_lyapunov_dim(lyap_exponents: np.ndarray) -> np.ndarray:
    lyap_exps_cum_sums = np.cumsum(lyap_exponents, axis=1)
    first_neg_sum = np.argmax(lyap_exps_cum_sums < 0, axis=1)
    lyap_dim = first_neg_sum + (
        lyap_exps_cum_sums[:, first_neg_sum - 1].diagonal()
        / np.abs(lyap_exponents[:, first_neg_sum].diagonal())
    )

    return np.clip(lyap_dim, a_min=0, a_max=None)


def get_holder_exponent(lyap_exponents: np.ndarray) -> np.ndarray:
    mle = get_maximum_lyap_exponent(lyap_exponents)
    return -np.log(0.99) / mle


def get_all_metrics(lyap_exponents: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    return {
        "PLE": get_sum_of_positive_lyap_exponents(lyap_exponents),
        "Lyapunov Time (base e, seconds)": get_lyap_time(lyap_exponents),
        "Lyapunov Time (base 10, seconds)": get_lyap_time(lyap_exponents, base=10),
        "Lyapunov Time (base 10, steps)": get_lyap_time(lyap_exponents, base=10, dt=dt),
        "MLE": get_maximum_lyap_exponent(lyap_exponents),
        "# Positive Exponents": get_count_positive_lyap_exponents(lyap_exponents),
        "SLE": get_sum_of_lyapunov_exponents(lyap_exponents),
        "Lyapunov Dimension": get_lyapunov_dim(lyap_exponents),
        "Holder Exponent": get_holder_exponent(lyap_exponents),
    }


def plot_lyapunov_exponents_and_metrics(
    lyap_exps_list: np.ndarray,
    save_loc: str,
    study_name: str,
    dt: float,
    x_values: Union[None, np.ndarray] = None,
    x_label: str = "",
    x_scale: str = "linear",
    y_scale: str = "linear",
    exponent_title: str = "",
    metric_title: str = "",
    n_cols: int = 3,
):
    # plot lyapunov exponents
    max_dims = np.max([lyap_exps.shape[-1] for lyap_exps in lyap_exps_list])
    lyap_exps_mean = [np.mean(lyap_exps, axis=0) for lyap_exps in lyap_exps_list]
    lyap_exps_mean = np.array(
        [
            np.pad(
                lyap_exps, (0, max_dims - lyap_exps.shape[0]), constant_values=np.nan
            )
            for lyap_exps in lyap_exps_mean
        ]
    )

    lyap_exps_std = [np.std(lyap_exps, axis=0) for lyap_exps in lyap_exps_list]
    lyap_exps_std = np.array(
        [
            np.pad(
                lyap_exps, (0, max_dims - lyap_exps.shape[0]), constant_values=np.nan
            )
            for lyap_exps in lyap_exps_std
        ]
    )

    x_values = np.arange(len(lyap_exps_mean)) if x_values is None else x_values

    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    axs.plot(x_values, lyap_exps_mean, color="tab:blue")
    for i in range(lyap_exps_mean.shape[-1]):
        axs.fill_between(
            x_values,
            (lyap_exps_mean[:, i] - 2 * lyap_exps_std[:, i]),
            (lyap_exps_mean[:, i] + 2 * lyap_exps_std[:, i]),
            color="tab:blue",
            alpha=0.2,
        )

    axs.axhline(0, color="red", ls="--")
    axs.set_xlabel(x_label)
    axs.set_ylabel("Exponent Value")
    axs.set_xscale(x_scale)
    axs.set_yscale("symlog")
    # axs.set_title(exponent_title)

    fig.tight_layout()
    os.makedirs(f"{save_loc}/{study_name}", exist_ok=True)
    plt.savefig(f"{save_loc}/{study_name}/exponents.png")
    plt.close()

    # plot Lyapunov metrics
    aggregated_metrics = {}
    for lyap_exps in lyap_exps_list:
        # get metrics
        lyap_metrics = get_all_metrics(lyap_exps, dt)

        # get metric statistics
        metric_stats = {
            key: get_series_statistics(val, return_tuple=False)
            for (key, val) in lyap_metrics.items()
        }
        for metric_key, metric_val in metric_stats.items():
            existing_metrics = aggregated_metrics.get(metric_key, [])
            existing_metrics.append(metric_val)
            aggregated_metrics[metric_key] = existing_metrics

    # plot aggregated metrics
    for metric_key, metric_values in aggregated_metrics.items():
        if metric_key not in ["MLE", "SLE"]:
            continue

        fig, axs = plt.subplots(1, 1, figsize=(3, 3))
        mean = [eps_metric_val["mean"] for eps_metric_val in metric_values]
        lower_ci = [eps_metric_val["ci"][0] for eps_metric_val in metric_values]
        upper_ci = [eps_metric_val["ci"][1] for eps_metric_val in metric_values]
        axs_x_values = np.arange(len(mean)) if x_values is None else x_values

        axs.plot(axs_x_values, np.zeros_like(axs_x_values), color="red", ls="--")
        axs.plot(axs_x_values, mean)
        axs.fill_between(
            axs_x_values,
            lower_ci,
            upper_ci,
            color="tab:blue",
            alpha=0.2,
        )
        axs.set_xlabel(x_label)
        axs.set_ylabel(metric_key)
        axs.set_xscale(x_scale)
        axs.set_yscale(y_scale)

        fig.tight_layout()
        plt.savefig(f"{save_loc}/{study_name}/{metric_key}.png")
        plt.close()

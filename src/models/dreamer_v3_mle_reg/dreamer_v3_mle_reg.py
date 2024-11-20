from typing import Optional, Type, Dict, Any, Tuple, Union, TypeVar

import numpy as np
import torch as th
from torch.distributions import Distribution, Independent
from torch.optim.adam import Adam as Adam
from torch.optim.optimizer import Optimizer as Optimizer
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from src.models.dreamer_v3 import Dreamer_V3
from src.models.dreamer_v3.utils import compute_lambda_values
from src.models.dreamer_v3.distributions import (
    TwoHotEncodingDistribution,
    BernoulliSafeMode,
)

Self_Dreamer_V3_Mle_Reg = TypeVar("Self_Dreamer_V3_Mle_Reg", bound="Dreamer_V3_Mle_Reg")


class Dreamer_V3_Mle_Reg(Dreamer_V3):
    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        rssm_sequence_length: int = 16,
        actor_critic_sequence_length: int = 64,
        batch_size: int = 256,
        free_nats: float = 1.0,
        recon_loss_weight: float = 1.0,
        dyn_loss_weight: float = 0.5,
        repr_loss_weight: float = 0.1,
        tau: float = 0.02,
        gamma: float = 0.997,
        lambda_val: float = 0.95,
        n_samples: int = 3,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = True,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        _init_setup_model: bool = True,
    ):
        self.n_samples = n_samples

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            rssm_sequence_length=rssm_sequence_length,
            actor_critic_sequence_length=actor_critic_sequence_length,
            batch_size=batch_size,
            free_nats=free_nats,
            recon_loss_weight=recon_loss_weight,
            dyn_loss_weight=dyn_loss_weight,
            repr_loss_weight=repr_loss_weight,
            tau=tau,
            gamma=gamma,
            lambda_val=lambda_val,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=supported_action_spaces,
            _init_setup_model=_init_setup_model,
        )

    def learn(
        self: Self_Dreamer_V3_Mle_Reg,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "Dreamer_V3_MLE_Reg",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> Self_Dreamer_V3_Mle_Reg:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def train(self, gradient_steps: int, batch_size: int) -> None:

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        obs_losses, reward_losses, continue_losses = [], [], []
        recon_losses, repr_losses, dyn_losses = [], [], []
        rssm_losses, actor_losses, critic_losses = [], [], []
        objective_losses, entropy_losses = [], []
        obs_std_losses, recurrent_state_std_losses = [], []
        div_reg_losses = []

        for _ in range(gradient_steps):
            # update target critic parameters
            polyak_update(
                self.critic.parameters(),
                self.target_critic.parameters(),
                self.tau,
            )

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                sequence_length=self.rssm_sequence_length,
                batch_size=batch_size,
            )

            # optimize world model
            (
                observation_loss,
                reward_loss,
                continue_loss,
                recon_loss,
                dyn_loss,
                repr_loss,
                rssm_loss,
                posteriors,
                recurrent_states,
            ) = self._optimize_world_model(
                replay_data.observations,
                replay_data.actions,
                replay_data.rewards,
                replay_data.is_first,
                replay_data.dones,
                batch_size,
            )

            # log world model metrics
            obs_losses.append(observation_loss)
            reward_losses.append(reward_loss)
            continue_losses.append(continue_loss)
            recon_losses.append(recon_loss)
            dyn_losses.append(dyn_loss)
            repr_losses.append(repr_loss)
            rssm_losses.append(rssm_loss)

            # optimize actor with MLE regularisation
            (
                objective,
                entropy,
                predicted_obs_std,
                recurrent_state_std,
                div_reg,
                actor_loss,
                imagined_trajectories,
                lambda_values,
                discount,
            ) = self._optimize_actor(
                posteriors,
                recurrent_states,
                replay_data.dones,
                batch_size,
            )
            objective_losses.append(objective)
            entropy_losses.append(entropy)
            obs_std_losses.append(predicted_obs_std)
            recurrent_state_std_losses.append(recurrent_state_std)
            div_reg_losses.append(div_reg)
            actor_losses.append(actor_loss)

            # optimize critic
            critic_loss = self._optimize_critic(
                imagined_trajectories,
                lambda_values,
                discount,
            )
            critic_losses.append(critic_loss)

            # reset everything
            self.policy.world_model_optimizer.zero_grad(set_to_none=True)
            self.policy.actor_optimizer.zero_grad(set_to_none=True)
            self.policy.critic_optimizer.zero_grad(set_to_none=True)

        # log world model losses
        self.logger.record("train_wm/observation_loss", np.mean(obs_losses))
        self.logger.record("train_wm/reward_loss", np.mean(reward_losses))
        self.logger.record("train_wm/continue_loss", np.mean(continue_losses))
        self.logger.record("train_wm/reconstruction_loss", np.mean(recon_losses))
        self.logger.record("train_wm/representation_loss", np.mean(repr_losses))
        self.logger.record("train_wm/dynamics_loss", np.mean(dyn_losses))
        self.logger.record("train_wm/total_loss", np.mean(rssm_losses))

        # log actor losses
        self.logger.record("train_actor/objective_loss", np.mean(objective_losses))
        self.logger.record("train_actor/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train_actor/obs_std_loss", np.mean(obs_std_losses))
        self.logger.record(
            "train_actor/recurrent_state_std_losses",
            np.mean(recurrent_state_std_losses),
        )
        self.logger.record("train_actor/div_reg_losses", np.mean(div_reg_losses))
        self.logger.record("train_actor/total_loss", np.mean(actor_losses))

        # log critic losses
        self.logger.record("train_critic/critic_loss", np.mean(critic_losses))

    def _optimize_actor(
        self,
        posteriors: th.Tensor,
        recurrent_states: th.Tensor,
        dones: th.Tensor,
        batch_size: int,
    ):
        """Optimise Dreamer V3 with MLE regularisation"""

        # behaviour learning
        imagined_prior = posteriors.detach().reshape(
            1, -1, self.policy.discrete_rep_total_size
        )
        recurrent_state = recurrent_states.detach().reshape(
            1, -1, self.policy.recurrent_state_size
        )

        # expand to n_samples (l) dimensions
        imagined_prior = imagined_prior.repeat([1, self.n_samples, 1])
        recurrent_state = recurrent_state.repeat([1, self.n_samples, 1])

        imagined_latent_state = th.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories = th.empty(
            self.actor_critic_sequence_length + 1,
            batch_size * self.rssm_sequence_length * self.n_samples,
            self.policy.latent_state_size,
            device=self.device,
        )
        imagined_trajectories[0] = imagined_latent_state
        imagined_actions = th.empty(
            self.actor_critic_sequence_length + 1,
            batch_size * self.rssm_sequence_length * self.n_samples,
            self.action_space.shape[-1],
            device=self.device,
        )
        actions = self.actor(imagined_latent_state.detach())[0]
        imagined_actions[0] = actions

        # Imagine trajectories in the latent space
        for i in range(1, self.actor_critic_sequence_length + 1):
            imagined_prior, recurrent_state = self.rssm.imagination(
                imagined_prior, recurrent_state, actions
            )
            imagined_prior = imagined_prior.view(
                1, -1, self.policy.discrete_rep_total_size
            )
            imagined_latent_state = th.cat((imagined_prior, recurrent_state), -1)
            imagined_trajectories[i] = imagined_latent_state
            actions = self.actor(imagined_latent_state.detach())[0]
            imagined_actions[i] = actions

        # Predict values, rewards and continues
        predicted_values = TwoHotEncodingDistribution(
            self.critic(imagined_trajectories), dims=1
        ).mean
        predicted_rewards = TwoHotEncodingDistribution(
            self.reward_model(imagined_trajectories), dims=1
        ).mean
        continues = Independent(
            BernoulliSafeMode(
                logits=self.continue_model(imagined_trajectories),
                validate_args=False,
            ),
            1,
            validate_args=False,
        ).mode
        true_done = (1 - dones).flatten().reshape(1, -1, 1)
        true_done = true_done.repeat([1, self.n_samples, 1])
        continues = th.cat((true_done, continues[1:]))

        # Estimate lambda-values
        lambda_values = compute_lambda_values(
            predicted_rewards[1:],
            predicted_values[1:],
            continues[1:] * self.gamma,
            lmbda=self.lambda_val,
        )

        # Compute the discounts to multiply the lambda values to
        with th.no_grad():
            discount = th.cumprod(continues * self.gamma, dim=0) / self.gamma

        self.policy.actor_optimizer.zero_grad(set_to_none=True)
        policies: Distribution = self.actor(imagined_trajectories.detach())[1]

        baseline = predicted_values[:-1]
        offset, invscale = self.moments(lambda_values)
        normed_lambda_values = (lambda_values - offset) / invscale
        normed_baseline = (baseline - offset) / invscale
        objective = normed_lambda_values - normed_baseline

        try:
            entropy = self.ent_coef * policies.entropy()
        except NotImplementedError:
            entropy = th.zeros_like(objective)

        # get predicted obs std
        predicted_obs: th.Tensor = self.decoder(imagined_trajectories)
        predicted_obs = th.stack(predicted_obs.chunk(self.n_samples, dim=1), dim=1)
        predicted_obs_std = predicted_obs.std(dim=1).mean(dim=-1, keepdim=True)

        # chunk discount, objective and entropy
        discount = th.stack(discount.chunk(self.n_samples, dim=1), dim=1)[:, 0]
        objective = th.stack(objective.chunk(self.n_samples, dim=1), dim=1)[:, 0]
        entropy = th.stack(entropy.chunk(self.n_samples, dim=1), dim=1)[:, 0]
        lambda_values = th.stack(lambda_values.chunk(self.n_samples, dim=1), dim=1)[
            :, 0
        ]

        # get imaged recurrent divergence
        imaged_recurrent: th.Tensor = imagined_trajectories.split(
            [self.policy.discrete_rep_total_size, self.policy.recurrent_state_size],
            dim=-1,
        )[1]
        all_imaged_recurrent = th.stack(
            imaged_recurrent.chunk(self.n_samples, dim=1), dim=1
        )
        recurrent_state_std = all_imaged_recurrent.std(dim=1).mean(dim=-1, keepdim=True)

        # get imagined logits
        imagined_logits = self.rssm._transition(imaged_recurrent)[0]
        all_imagined_logits = th.stack(
            imagined_logits.chunk(self.n_samples, dim=1), dim=1
        )
        all_imagined_logits = all_imagined_logits.view(
            *all_imagined_logits.shape[:-1],
            self.policy.discrete_rep_size,
            self.policy.discrete_rep_n_classes,
        )

        # get divergence regularisation
        div_reg = predicted_obs_std + recurrent_state_std

        # get imagined trajectories
        imagined_trajectories = th.stack(
            imagined_trajectories.chunk(self.n_samples, dim=1), dim=1
        )[:, 0]

        # calculate loss
        actor_loss = -th.mean(
            discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1])
            - div_reg[1:]
        )
        actor_loss.backward()
        th.nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
        self.policy.actor_optimizer.step()

        return (
            -objective.detach().mean().item(),
            -entropy.detach().mean().item(),
            predicted_obs_std.detach().mean().item(),
            recurrent_state_std.detach().mean().item(),
            div_reg.detach().mean().item(),
            actor_loss.detach().item(),
            imagined_trajectories.detach(),
            lambda_values.detach(),
            discount.detach(),
        )

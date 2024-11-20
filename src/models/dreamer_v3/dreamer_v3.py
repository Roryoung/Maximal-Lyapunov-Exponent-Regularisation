"""
Implementation of Dreamer V3 based on the Electric Sheep implementation [1]

[1]: https://github.com/Eclectic-Sheep/sheeprl/tree/main/sheeprl/algos/dreamer_v3
"""

from typing import Optional, Type, List, Dict, Any, Tuple, Union, ClassVar, TypeVar

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Distribution, Independent
from torch.distributions.kl import kl_divergence
from torch.optim.adam import Adam as Adam
from torch.optim.optimizer import Optimizer as Optimizer
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

import io
import pathlib
import warnings
from stable_baselines3.common.utils import check_for_correct_spaces, get_system_info
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from src.models.dreamer_v3.utils import compute_lambda_values
from src.models.dreamer_v3.models import Encoder, Decoder, Actor, Moments, World_Model
from src.models.dreamer_v3.dreamer_v3_policy import Dreamer_V3_Policy
from src.models.dreamer_v3.replay_buffer import Sequential_Replay_Buffer
from src.models.dreamer_v3.distributions import (
    SymlogDistribution,
    TwoHotEncodingDistribution,
    BernoulliSafeMode,
    OneHotCategoricalStraightThroughValidateArgs,
)


Self_Dreamer_V3 = TypeVar("Self_Dreamer_V3", bound="Dreamer_V3")


class Dreamer_V3(OffPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": Dreamer_V3_Policy,
    }
    policy: Dreamer_V3_Policy
    encoder: Encoder
    decoder: Decoder
    reward_model: nn.Module
    continue_model: nn.Module
    world_model: World_Model
    actor: Actor
    critic: nn.Module
    target_critic: nn.Module
    replay_buffer: Sequential_Replay_Buffer

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        rssm_sequence_length: int = 64,
        actor_critic_sequence_length: int = 16,
        batch_size: int = 256,
        free_nats: float = 1.0,
        recon_loss_weight: float = 1.0,
        dyn_loss_weight: float = 0.5,
        repr_loss_weight: float = 0.1,
        tau: float = 0.02,
        gamma: float = 0.99,
        lambda_val: float = 0.95,
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
        self.rssm_sequence_length = rssm_sequence_length
        self.actor_critic_sequence_length = actor_critic_sequence_length
        self.free_nats = free_nats
        self.recon_loss_weight = recon_loss_weight
        self.dyn_loss_weight = dyn_loss_weight
        self.repr_loss_weight = repr_loss_weight
        self.lambda_val = lambda_val
        self.ent_coef = 3e-4

        if isinstance(policy_kwargs, Dict):
            policy_kwargs.pop("net_arch", None)
            policy_kwargs["num_envs"] = env.num_envs

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=None,
            replay_buffer_class=Sequential_Replay_Buffer,
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
            sde_support=False,
            supported_action_spaces=supported_action_spaces,
        )

        self.moments = Moments(
            decay=0.99,
            max_=1.0,
            percentile_low=0.05,
            percentile_high=0.95,
        ).to(self.device)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.policy.num_envs = self.n_envs
        self.policy.init_states()

    def _create_aliases(self) -> None:
        self.encoder = self.policy.world_model.encoder
        self.decoder = self.policy.world_model.decoder
        self.reward_model = self.policy.world_model.reward_model
        self.continue_model = self.policy.world_model.continue_model
        self.rssm = self.policy.world_model.rssm
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.target_critic = self.policy.target_critic

    def _store_transition(
        self,
        replay_buffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super()._store_transition(
            replay_buffer, buffer_action, new_obs, reward, dones, infos
        )
        if np.any(dones):
            self.policy.init_states(dones)

    def learn(
        self: Self_Dreamer_V3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "Dreamer_V3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> Self_Dreamer_V3:
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

            # optimize actor
            (actor_loss, imagined_trajectories, lambda_values, discount) = (
                self._optimize_actor(
                    posteriors,
                    recurrent_states,
                    replay_data.dones,
                    batch_size,
                )
            )
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

        # log recurrent sate space model losses
        self.logger.record("train/observation_loss", np.mean(obs_losses))
        self.logger.record("train/reward_loss", np.mean(reward_losses))
        self.logger.record("train/continue_loss", np.mean(continue_losses))
        self.logger.record("train/reconstruction_loss", np.mean(recon_losses))
        self.logger.record("train/representation_loss", np.mean(repr_losses))
        self.logger.record("train/dynamics_loss", np.mean(dyn_losses))
        self.logger.record("train/rssm_loss", np.mean(rssm_losses))

        # log actor critic losses
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def _optimize_world_model(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        is_first: th.Tensor,
        dones: th.Tensor,
        batch_size: int,
    ):
        # Given how the environment interaction works, we remove the last actions
        # and add the first one as the zero action
        batch_actions = th.cat(
            (th.zeros_like(actions[:1]), actions[:-1]),
            dim=0,
        )

        # Dynamics Learning
        recurrent_state = th.zeros(
            1, batch_size, self.policy.recurrent_state_size, device=self.device
        )
        posterior = th.zeros(
            1,
            batch_size,
            self.policy.discrete_rep_size,
            self.policy.discrete_rep_n_classes,
            device=self.device,
        )
        recurrent_states = th.empty(
            self.rssm_sequence_length,
            batch_size,
            self.policy.recurrent_state_size,
            device=self.device,
        )
        priors_logits = th.empty(
            self.rssm_sequence_length,
            batch_size,
            self.policy.discrete_rep_total_size,
            device=self.device,
        )
        posteriors = th.empty(
            self.rssm_sequence_length,
            batch_size,
            self.policy.discrete_rep_size,
            self.policy.discrete_rep_n_classes,
            device=self.device,
        )
        posteriors_logits = th.empty(
            self.rssm_sequence_length,
            batch_size,
            self.policy.discrete_rep_total_size,
            device=self.device,
        )

        # Embed observations from the environment
        embedded_obs = self.encoder(observations)
        is_first[0] = th.ones_like(is_first[0])

        for i in range(0, self.rssm_sequence_length):
            recurrent_state, posterior, _, posterior_logits, prior_logits = (
                self.rssm.dynamic(
                    posterior,
                    recurrent_state,
                    batch_actions[i : i + 1],
                    embedded_obs[i : i + 1],
                    is_first[i : i + 1],
                )
            )
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits
            posteriors[i] = posterior
            posteriors_logits[i] = posterior_logits

        latent_states = th.cat(
            (posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1
        )

        # Compute predictions for the observations
        reconstructed_obs: th.Tensor = self.decoder(latent_states)

        # Compute the distribution over the reconstructed observations
        obs_dist = SymlogDistribution(
            reconstructed_obs,
            dims=len(reconstructed_obs.shape[2:]),
        )

        # Compute the distribution over the rewards
        reward_dist = TwoHotEncodingDistribution(
            self.reward_model(latent_states), dims=1
        )

        # Compute the distribution over the terminal steps, if required
        continue_dist = Independent(
            BernoulliSafeMode(
                logits=self.continue_model(latent_states),
                validate_args=False,
            ),
            1,
            validate_args=False,
        )
        continue_targets = 1 - dones

        # Reshape posterior and prior logits to shape [T, B, 32, 32]
        priors_logits = priors_logits.view(
            *priors_logits.shape[:-1],
            self.policy.discrete_rep_size,
            self.policy.discrete_rep_n_classes,
        )
        posteriors_logits = posteriors_logits.view(
            *posteriors_logits.shape[:-1],
            self.policy.discrete_rep_size,
            self.policy.discrete_rep_n_classes,
        )

        self.policy.world_model_optimizer.zero_grad(set_to_none=True)

        # compute observation loss
        observation_loss = -obs_dist.log_prob(observations)

        # compute reward loss
        reward_loss = -reward_dist.log_prob(rewards)

        # compute continue loss
        continue_loss: th.Tensor = -continue_dist.log_prob(continue_targets)

        # compute reconstruction loss
        recon_loss = self.recon_loss_weight * (
            observation_loss + reward_loss + continue_loss
        )

        # dynamics loss
        dyn_loss = kl_divergence(
            Independent(
                OneHotCategoricalStraightThroughValidateArgs(
                    logits=posteriors_logits.detach(), validate_args=False
                ),
                1,
                validate_args=False,
            ),
            Independent(
                OneHotCategoricalStraightThroughValidateArgs(
                    logits=priors_logits, validate_args=False
                ),
                1,
                validate_args=False,
            ),
        )
        free_nats = th.full_like(dyn_loss, self.free_nats)
        dyn_loss = self.dyn_loss_weight * th.maximum(dyn_loss, free_nats)

        # representation loss
        repr_loss = kl_divergence(
            Independent(
                OneHotCategoricalStraightThroughValidateArgs(
                    logits=posteriors_logits, validate_args=False
                ),
                1,
                validate_args=False,
            ),
            Independent(
                OneHotCategoricalStraightThroughValidateArgs(
                    logits=priors_logits.detach(), validate_args=False
                ),
                1,
                validate_args=False,
            ),
        )
        repr_loss = self.repr_loss_weight * th.maximum(repr_loss, free_nats)

        rssm_loss = (recon_loss + dyn_loss + repr_loss).mean()
        rssm_loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), 1000)
        self.policy.world_model_optimizer.step()

        return (
            observation_loss.mean().item(),
            reward_loss.mean().item(),
            continue_loss.mean().item(),
            recon_loss.mean().item(),
            dyn_loss.mean().item(),
            repr_loss.mean().item(),
            rssm_loss.item(),
            posteriors.detach(),
            recurrent_states.detach(),
        )

    def _optimize_actor(
        self,
        posteriors: th.Tensor,
        recurrent_states: th.Tensor,
        dones: th.Tensor,
        batch_size: int,
    ):
        # behaviour learning
        imagined_prior = posteriors.detach().reshape(
            1, -1, self.policy.discrete_rep_total_size
        )
        recurrent_state = recurrent_states.detach().reshape(
            1, -1, self.policy.recurrent_state_size
        )
        imagined_latent_state = th.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories = th.empty(
            self.actor_critic_sequence_length + 1,
            batch_size * self.rssm_sequence_length,
            self.policy.latent_state_size,
            device=self.device,
        )
        imagined_trajectories[0] = imagined_latent_state
        imagined_actions = th.empty(
            self.actor_critic_sequence_length + 1,
            batch_size * self.rssm_sequence_length,
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

        actor_loss = -th.mean(
            discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1])
        )
        actor_loss.backward()
        th.nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
        self.policy.actor_optimizer.step()

        return (
            actor_loss.detach().item(),
            imagined_trajectories.detach(),
            lambda_values.detach(),
            discount.detach(),
        )

    def _optimize_critic(
        self,
        imagined_trajectories: th.Tensor,
        lambda_values: th.Tensor,
        discount: th.Tensor,
    ) -> float:
        # get value distributions
        value_dist = TwoHotEncodingDistribution(
            self.critic(imagined_trajectories[:-1]), dims=1
        )
        predicted_target_values = TwoHotEncodingDistribution(
            self.target_critic(imagined_trajectories[:-1]), dims=1
        ).mean

        # Critic optimization. Eq. 10 in the paper
        self.policy.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss = -value_dist.log_prob(lambda_values.detach())
        critic_loss = critic_loss - value_dist.log_prob(
            predicted_target_values.detach()
        )
        critic_loss = th.mean(critic_loss * discount[:-1].squeeze(-1))
        critic_loss.backward()
        th.nn.utils.clip_grad_norm_(self.critic.parameters(), 100)
        self.policy.critic_optimizer.step()

        return critic_loss.detach().item()

    def _excluded_save_params(self) -> List[str]:
        return_val = super()._excluded_save_params() + [
            "encoder",
            "decoder",
            "reward_model",
            "continue_model",
            "world_model",
            "rssm",
            "actor",
            "critic",
            "target_critic",
            "policy.actions",
            "policy.recurrent_state",
            "policy.stochastic_state",
        ]  # noqa: RUF005
        return_val.remove("replay_buffer")
        return return_val

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = [
            "policy",
            "policy.world_model_optimizer",
            "policy.actor_optimizer",
            "policy.critic_optimizer",
        ]
        return state_dicts, []

    @classmethod
    def load(  # noqa: C901
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ):
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        if "actions" in params["policy"]:
            params["policy"].pop("actions")

        if "recurrent_state" in params["policy"]:
            params["policy"].pop("recurrent_state")

        if "stochastic_state" in params["policy"]:
            params["policy"].pop("stochastic_state")

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if (
                "net_arch" in data["policy_kwargs"]
                and len(data["policy_kwargs"]["net_arch"]) > 0
            ):
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(
                    saved_net_arch[0], dict
                ):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if (
            "policy_kwargs" in kwargs
            and kwargs["policy_kwargs"] != data["policy_kwargs"]
        ):
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError(
                "The observation_space and action_space were not given, can't verify new environments"
            )

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(
                data[key]
            )  # pytype: disable=unsupported-operands

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(
                env, data["observation_space"], data["action_space"]
            )
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # pytype: disable=not-instantiable,wrong-keyword-args
        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )
        # pytype: enable=not-instantiable,wrong-keyword-args

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(
                e
            ) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]  # pytype: disable=attribute-error
        return model

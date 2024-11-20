"""
Actor, Critic and World Models used for Dreamer V3 and Dreamer V3 + MLE regularisation.
This work is based on the Electric Sheep implementation of Dreamer V3 [1]

[1]: https://github.com/Eclectic-Sheep/sheeprl/tree/main/sheeprl/algos/dreamer_v3
"""

from typing import Optional, Type, Dict, Any, Sequence
import copy

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
import numpy as np
import torch as th
from torch import nn
from torch.optim.adam import Adam as Adam
from torch.optim.optimizer import Optimizer as Optimizer
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy


from src.models.dreamer_v3.utils import (
    symexp,
    make_mlp,
    init_weights,
    uniform_init_weights,
)
from src.models.dreamer_v3.models import (
    Encoder,
    Decoder,
    RSSM,
    Recurrent_Model,
    Actor,
    World_Model,
)


class Dreamer_V3_Policy(BasePolicy):
    world_model: World_Model
    actor: Actor
    critic: nn.Module
    target_critic: nn.Module
    world_model_optimizer: th.optim.Adam
    actor_optimizer: th.optim.Adam
    critic_optimizer: th.optim.Adam

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        num_envs: int,
        activation_fn: Type[nn.Module] = nn.SiLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        reward_bins: int = 255,
        critic_bins: int = 255,
        recurrent_state_size: int = 4096,
        discrete_rep_size: int = 32,
        discrete_rep_n_classes: int = 32,
        layer_norm: bool = True,
        net_width: int = 1024,
        net_depth: int = 5,
        representation_width: int = 1024,
        dynamics_width: int = 1024,
        uni_mix: int = 0.01,
        hafner_initialization: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        self.num_envs = num_envs

        # save latent space dimensions
        self.recurrent_state_size = recurrent_state_size
        self.discrete_rep_size = discrete_rep_size
        self.discrete_rep_n_classes = discrete_rep_n_classes
        self.discrete_rep_total_size = discrete_rep_size * discrete_rep_n_classes
        self.latent_state_size = self.discrete_rep_total_size + recurrent_state_size
        self.reward_bins = reward_bins
        self.critic_bins = critic_bins

        # save network parameters
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm
        self.network_width = net_width
        self.network_depth = net_depth
        self.representation_width = representation_width
        self.dynamics_width = dynamics_width
        self.uni_mix = uni_mix
        self.hafner_initialization = hafner_initialization

        # build all models
        self._build(lr_schedule)
        self.init_states()

    def _build(self, lr_schedule: Schedule) -> None:
        # build models
        self._build_world_model()
        self._build_actor()
        self._build_critic()
        self.target_critic = copy.deepcopy(self.critic)

        # build optimizers
        self.world_model_optimizer = th.optim.Adam(
            self.world_model.parameters(), lr=1e-4, eps=1e-8
        )
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=8e-5, eps=1e-5)
        self.critic_optimizer = th.optim.Adam(
            self.critic.parameters(), lr=8e-5, eps=1e-5
        )

    def _build_world_model(self):
        # encoder: (x -> 1024)
        encoder = Encoder(
            obs_dim=self.observation_space.shape[0],
            net_arch=[self.network_width] * self.network_depth,
            activation_fn=self.activation_fn,
            layer_norm=self.layer_norm,
        )

        # decoder: (h + z -> x)
        decoder = Decoder(
            output_dim=self.observation_space.shape[0],
            latent_state_size=self.latent_state_size,
            net_arch=[self.network_width] * self.network_depth,
            activation_fn=self.activation_fn,
            layer_norm=self.layer_norm,
        )

        # sequence model: (a + z + h -> h)
        recurrent_model = Recurrent_Model(
            input_size=self.action_space.shape[0] + self.discrete_rep_total_size,
            recurrent_state_size=self.recurrent_state_size,
            n_hidden_units=self.network_width,
            activation_fn=self.activation_fn,
            layer_norm=self.layer_norm,
        )

        # representation_model: h + encoder(obs) -> z
        representation_model = make_mlp(
            net_arch=[
                self.recurrent_state_size + self.network_width,
                self.representation_width,
            ],
            activation_fn=self.activation_fn,
            layer_norm=self.layer_norm,
            output_dim=self.discrete_rep_total_size,
        )

        # dynamics predictor: (h -> z)
        dynamics_predictor = make_mlp(
            net_arch=[self.recurrent_state_size, self.dynamics_width],
            activation_fn=self.activation_fn,
            layer_norm=self.layer_norm,
            output_dim=self.discrete_rep_total_size,
        )

        # recurrent state space model
        rssm = RSSM(
            recurrent_model=recurrent_model.apply(init_weights),
            representation_model=representation_model.apply(init_weights),
            dynamics_predictor=dynamics_predictor.apply(init_weights),
            discrete_rep_n_classes=self.discrete_rep_n_classes,
            uni_mix=self.uni_mix,
        )

        # reward model: (h + z -> r_bins)
        reward_model = make_mlp(
            net_arch=[self.latent_state_size]
            + [self.network_width] * self.network_depth,
            layer_norm=self.layer_norm,
            activation_fn=self.activation_fn,
            output_dim=self.reward_bins,
        )

        # continue model: (h + z -> 1)
        continue_model = make_mlp(
            net_arch=[self.latent_state_size]
            + [self.network_width] * self.network_depth,
            layer_norm=self.layer_norm,
            activation_fn=self.activation_fn,
            output_dim=1,
        )

        self.world_model = World_Model(
            encoder.apply(init_weights),
            rssm,
            decoder.apply(init_weights),
            reward_model.apply(init_weights),
            continue_model.apply(init_weights),
        )

        if self.hafner_initialization:
            self.world_model.rssm.dynamics_predictor[-1].apply(
                uniform_init_weights(1.0)
            )
            self.world_model.rssm.representation_model[-1].apply(
                uniform_init_weights(1.0)
            )
            self.world_model.reward_model[-1].apply(uniform_init_weights(0.0))
            self.world_model.continue_model[-1].apply(uniform_init_weights(1.0))
            self.world_model.decoder.head.apply(uniform_init_weights(1.0))

    def _build_actor(self):
        # actor (h + z -> a)
        self.actor = Actor(
            self.latent_state_size,
            self.action_space.shape[0],
            net_arch=[self.network_width] * self.network_depth,
            activation_fn=self.activation_fn,
            layer_norm=self.layer_norm,
        )
        self.actor.apply(init_weights)
        if self.hafner_initialization:
            self.actor.head.apply(uniform_init_weights(1.0))

    def _build_critic(self):
        # critic (h + z -> c_bins)
        self.critic = make_mlp(
            net_arch=[self.latent_state_size]
            + [self.network_width] * self.network_depth,
            layer_norm=self.layer_norm,
            activation_fn=self.activation_fn,
            output_dim=self.critic_bins,
        )
        self.critic.apply(init_weights)
        if self.hafner_initialization:
            self.critic[-1].apply(uniform_init_weights(1.0))

    def to(self, device):
        self = super().to(device)
        self.actions = self.actions.to(device)
        self.recurrent_state = self.recurrent_state.to(device)
        self.stochastic_state = self.stochastic_state.to(device)
        return self

    @th.no_grad()
    def init_states(self, reset_envs: Optional[Sequence[int]] = None) -> None:
        """Initialize the states and the actions for the ended environments.

        Args:
            reset_envs (Optional[Sequence[int]], optional): which environments' states to reset.
                If None, then all environments' states are reset.
                Defaults to None.
        """
        if reset_envs is None or len(reset_envs) == 0:
            self.actions = th.zeros(
                1,
                self.num_envs,
                np.sum(self.action_space.shape[0]),
                device=self.device,
            )

            self.recurrent_state = th.tanh(
                th.zeros(
                    1,
                    self.num_envs,
                    self.recurrent_state_size,
                    device=self.device,
                )
            )

            self.stochastic_state = self.world_model.rssm._transition(
                self.recurrent_state, sample_state=False
            )[1].reshape(1, self.num_envs, -1)

        else:
            self.actions[:, reset_envs] = th.zeros_like(self.actions[:, reset_envs])
            self.recurrent_state[:, reset_envs] = th.tanh(
                th.zeros_like(self.recurrent_state[:, reset_envs])
            )
            self.stochastic_state[:, reset_envs] = self.world_model.rssm._transition(
                self.recurrent_state[:, reset_envs], sample_state=False
            )[1].reshape(1, len(reset_envs), -1)

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        if len(observation.shape) == 2:
            observation = observation[None, ...]

        if deterministic:
            return self.get_greedy_action(observation.float(), deterministic=True)

        return self.get_exploration_action(observation.float())

    def get_exploration_action(
        self, obs: th.Tensor, mask: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """
        Return the actions with a certain amount of noise for exploration.

        Args:
            obs (Dict[str, Tensor]): the current observations.
            mask (Dict[str, Tensor], optional): the mask of the actions.
                Default to None.

        Returns:
            The actions the agent has to perform.
        """
        actions = self.get_greedy_action(obs, mask=mask, deterministic=False)
        if self.actor.explore_amount > 0:
            expl_actions = self.actor.add_exploration_noise(actions, mask=mask)[None, :]
            self.actions = expl_actions
            return expl_actions

        return actions

    def get_greedy_action(
        self,
        obs: th.Tensor,
        mask: Optional[th.Tensor] = None,
        deterministic: bool = False,
    ) -> th.Tensor:
        """
        Return the greedy actions.

        Args:
            obs (Dict[str, Tensor]): the current observations.

        Returns:
            The actions the agent has to perform.
        """
        embedded_obs = self.world_model.encoder(obs)
        self.recurrent_state = self.world_model.rssm.recurrent_model(
            th.cat((self.stochastic_state, self.actions), -1), self.recurrent_state
        )

        _, self.stochastic_state = self.world_model.rssm._representation(
            self.recurrent_state,
            embedded_obs,
            deterministic=deterministic,
        )
        self.stochastic_state = self.stochastic_state.view(
            *self.stochastic_state.shape[:-2], self.discrete_rep_total_size
        )

        actions = self.actor(
            th.cat((self.stochastic_state, self.recurrent_state), -1),
            mask,
            deterministic=deterministic,
        )[0]
        self.actions = actions
        return actions

    def trajectory_predict_n_steps(
        self,
        obs_traj: th.Tensor,
        action_traj: th.Tensor,
        n_steps: int,
        future_action_traj: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        batch_size = obs_traj.shape[1]

        obs_traj = obs_traj.float().to(self.device)
        action_traj = action_traj.float().to(self.device)

        # get initial action recurrent state and posterior
        recurrent_state = th.zeros(
            1,
            batch_size,
            self.recurrent_state_size,
            device=self.device,
        )
        posterior = th.zeros(
            1,
            batch_size,
            self.discrete_rep_size,
            self.discrete_rep_n_classes,
            device=self.device,
        )
        is_first = th.zeros(obs_traj.shape[:-1] + (1,)).to(self.device)
        is_first[0] = th.ones_like(is_first[0])

        embedded_obs = self.world_model.encoder(obs_traj)
        batch_actions = th.cat(
            (th.zeros_like(action_traj[:1]), action_traj[:-1]),
            dim=0,
        )

        for i in range(embedded_obs.shape[0]):
            recurrent_state, posterior = self.world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                embedded_obs[i : i + 1],
                is_first[i : i + 1],
            )[:2]

        imagined_prior = posterior.reshape(1, batch_size, self.discrete_rep_total_size)
        imagined_latent_state = th.cat((imagined_prior, recurrent_state), -1)
        current_obs = obs_traj[-1]

        # predict over n steps
        future_obs_traj = th.empty(n_steps, batch_size, obs_traj.shape[-1])
        for i in range(n_steps):
            # update observation trajectory
            future_obs_traj[i] = current_obs

            # compute next action
            if future_action_traj is None:
                actions = self.actor(imagined_latent_state, deterministic=True)[0]
            else:
                actions = future_action_traj[i : i + 1]

            # conduct one step prediction
            imagined_prior, recurrent_state = self.world_model.rssm.imagination(
                imagined_prior,
                recurrent_state,
                actions,
            )

            # reconstruct full latent space
            imagined_prior = imagined_prior.view(1, -1, self.discrete_rep_total_size)
            imagined_latent_state = th.cat((imagined_prior, recurrent_state), -1)

            # reconstruct imagined obs
            current_obs = symexp(self.world_model.decoder(imagined_latent_state)[:, 0])

        return future_obs_traj

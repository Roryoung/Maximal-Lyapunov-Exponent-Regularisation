from typing import Any, Dict, List, Optional, Type, Union

import torch as th
from torch import Tensor, nn
from gymnasium import spaces

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from torch.optim.adam import Adam as Adam
from torch.optim.optimizer import Optimizer as Optimizer


class Actor(BasePolicy):
    def forward(self, obs: th.Tensor) -> th.Tensor:
        batch_size = obs.shape[0]
        return th.zeros((batch_size,) + self.action_space.shape)

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        return self(observation)


class No_Action_Policy(BasePolicy):
    actor: Actor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        use_sde: bool = False,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
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

        self.actor = Actor(
            observation_space=observation_space,
            action_space=action_space,
        ).to(self.device)

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: Tensor, deterministic: bool = False) -> Tensor:
        return self.actor(observation)


MlpPolicy = No_Action_Policy

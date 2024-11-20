from typing import Any, ClassVar, Dict, Tuple, Type, TypeVar, Union
from gymnasium.spaces.space import Space

from torch import device
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from src.models.no_action.no_action_policy import Actor, MlpPolicy, No_Action_Policy

SelfNoAction = TypeVar("SelfNoAction", bound="No_Action")


class No_Action(OffPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
    }
    policy: No_Action_Policy
    actor: Actor

    def __init__(
        self,
        policy: Union[str, type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Union[ActionNoise, None] = None,
        replay_buffer_class: Union[type[ReplayBuffer], None] = None,
        replay_buffer_kwargs: Union[Dict[str, Any], None] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Union[Dict[str, Any], None] = None,
        stats_window_size: int = 100,
        tensorboard_log: Union[str, None] = None,
        verbose: int = 0,
        device: Union[device, str] = "auto",
        support_multi_env: bool = True,
        monitor_wrapper: bool = True,
        seed: Union[int, None] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        supported_action_spaces: Union[Tuple[type[Space], ...], None] = None,
        _init_setup_model=False,
    ):
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
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
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
            sde_support=sde_support,
            supported_action_spaces=supported_action_spaces,
        )
        self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

    def train(self, gradient_steps: int, batch_size: int) -> None:
        pass

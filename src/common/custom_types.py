from dataclasses import dataclass
from typing import Tuple, Union, List

from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass
class Env_Spec:
    env_source_name: str
    env_name: Union[str, Tuple[str, str]]
    task_kwargs: dict
    env_kwargs: dict
    n_train_envs: int
    n_eval_envs: int


@dataclass
class Agent_Spec:
    agent_name: str
    agent_cls: BaseAlgorithm
    agent_kwargs: dict
    n_eval_steps: int = 100_000
    save_steps: int = 1
    save_agent: bool = True
    load_agent: bool = True


@dataclass
class Trial_Spec:
    trial_name: str
    env_spec: Env_Spec
    agent_spec: Agent_Spec
    save_loc: str
    n_steps: int
    n_seeds: int


@dataclass
class Series_Spec:
    series_name: str
    series_values: List[float]
    series_save_values: List[str]

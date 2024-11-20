import os
from typing import Union, List, Dict, Optional
from abc import ABC, abstractmethod

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.base_class import BaseAlgorithm

from src.common.custom_types import Series_Spec
from src.common.utils import machine_format


class Base_Collate(ABC):
    short_agent_names = {"dreamer_v3": "dr3", "dreamer_v3_mle_reg": "mle dr3"}

    def __init__(
        self,
        results_dir: str,
        agent_list: List[str],
        env_source_name: str,
        env_list: Union[List[str], List[tuple[str, str]]],
        save_step: str,
        n_seeds: int,
        verbose: bool = True,
        series_spec: Optional[Series_Spec] = None,
    ) -> None:
        # experiment locations
        self.results_dir = results_dir
        self.agent_list = agent_list
        self.env_source_name = env_source_name
        self.env_list = env_list
        self.series_spec = series_spec

        # seed specific data
        self.save_step = save_step
        self.n_seeds = n_seeds

        # general
        self.verbose = verbose

    def collate(self):
        """Get all results and plot figures"""
        if self.save_step != "final":
            return

        all_results = self._get_all_results()
        if self.series_spec is None:
            self._collate_results(all_results)
        else:
            self._collate_series_results(all_results)

    def _get_all_results(self):
        """Loop over all environments and get env results"""

        all_results = {}
        for env in self.env_list:
            all_results["_".join(env)] = self._get_env_results(env)
        return all_results

    def _get_env_results(self, env: str) -> Dict:
        """For a given environment get all agent results"""

        env_results = {}
        for agent in self.agent_list:
            short_agent_name = self.short_agent_names.get(agent, agent)
            env_results[short_agent_name] = self._get_env_agent_results(env, agent)
        return env_results

    def _get_env_agent_results(self, env: str, agent: str) -> List:
        """For a given environment and agent get all seed results"""

        agent_results = {}
        if self.series_spec is not None:
            for series_value in self.series_spec.series_save_values:
                seed_results = []
                for seed_i in range(self.n_seeds):
                    seed_results.append(
                        self._get_seed_results(env, agent, seed_i, series_value)
                    )
                agent_results[series_value] = seed_results
            return agent_results
        else:
            agent_results = []
            for seed_i in range(self.n_seeds):
                agent_results.append(self._get_seed_results(env, agent, seed_i, None))
            return agent_results

    def _get_seed_results(
        self, env: str, agent: str, seed_i: int, series_value: Optional[str]
    ):
        """Get seed specific results"""

        # get eval save location
        if self.env_source_name == "dmc":
            env_string = f"{self.env_source_name}_" + "_".join(env)
        else:
            env_string = f"{self.env_source_name}_{env}"

        if series_value is None:
            seed_save_loc = f"{self.results_dir}/{agent}/{env_string}/seed={seed_i}"
        else:
            seed_save_loc = (
                f"{self.results_dir}/{agent}/{env_string}/{series_value}/seed={seed_i}"
            )
        seed_ckpt_results_dir = f"{seed_save_loc}/train_eval"

        if not os.path.exists(seed_ckpt_results_dir):
            return {}

        # get all saved data during training
        all_int_save_names = os.listdir(seed_ckpt_results_dir)
        int_save_steps = [machine_format(str_num) for str_num in all_int_save_names]
        ordering = np.argsort(int_save_steps).astype(int)
        sorted_all_int_save_names = np.array(all_int_save_names)[ordering].tolist()
        sorted_int_save_steps = np.array(int_save_steps)[ordering].tolist()

        if self.save_step == "final":
            sorted_all_int_save_names = sorted_all_int_save_names[-1:]
            sorted_int_save_steps = sorted_int_save_steps[-1:]

        # collate intermediate results
        ckpt_results = {}
        for ckpt_results_dir, ckpt_step in zip(
            sorted_all_int_save_names, sorted_int_save_steps
        ):
            results_file_loc = f"{seed_ckpt_results_dir}/{ckpt_results_dir}"
            ckpt_results[ckpt_step] = self._get_eval_function_results(results_file_loc)
        return ckpt_results

    @abstractmethod
    def _get_eval_function_results(self, save_loc: str):
        pass

    @abstractmethod
    def _collate_results(self, all_results):
        pass

    @abstractmethod
    def _collate_series_results(self, all_results):
        pass


class Base_Eval(ABC):
    def __init__(
        self,
        model: BaseAlgorithm,
        eval_env: Union[VecEnv, VecNormalize],
        env_source_name: str,
        env_name: Union[str, tuple[str, str]],
        results_dir: str,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.eval_env = eval_env
        self.results_dir = results_dir
        self.env_source_name = env_source_name
        self.env_name = env_name
        self.verbose = verbose
        self.dt = 1
        self.max_time = 1000

    def eval(
        self,
        save_loc: str,
        eval_plot: bool = True,
        eval_metric: bool = True,
    ) -> None:
        save_loc = f"{save_loc}/{self.get_eval_name()}"
        os.makedirs(save_loc, exist_ok=True)
        print(f" -{self.get_eval_name()}")

        if eval_plot:
            self._plt_all_figs(save_loc)

        if eval_metric:
            self._save_metrics(save_loc)

    @classmethod
    def get_collate_results_class(cls) -> Union[Base_Collate, None]:
        return None

    @classmethod
    @abstractmethod
    def get_eval_name(self):
        pass

    @abstractmethod
    def _plt_all_figs(self, save_loc: str) -> None:
        pass

    @abstractmethod
    def _save_metrics(self, save_loc: str) -> None:
        pass

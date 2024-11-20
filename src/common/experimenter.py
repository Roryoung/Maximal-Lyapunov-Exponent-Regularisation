"""This module outlines the general experiment class used to run multiple trials"""

import os
from typing import Union, List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import yaml
import argparse

import numpy as np
import torch as th
import seaborn as sns
from stable_baselines3 import SAC, TD3
from dm_control.mujoco import Physics

from src.models import *
from src.evaluation.base_eval import Base_Eval
from src.common.trainer import Trainer
from src.common.custom_types import Trial_Spec, Series_Spec
from src.common.utils import (
    fancy_print,
    underline_print,
    bold_print,
    print_hbar,
    human_format,
    str_to_act_fn,
    sigmoid,
)

sns.set_theme(style="darkgrid")


def env_name_type(input_string: str) -> Union[str, tuple]:
    """Correctly formats environments as tuples when `.` is used"""
    try:
        if "." in input_string:
            split_input_string = tuple(input_string.split("."))
            assert len(split_input_string) == 2
            return split_input_string

        return input_string
    except Exception as exc:
        raise argparse.ArgumentTypeError("Error") from exc


class Base_Experimenter(ABC):
    """General class used to run multiple trials"""

    # All storage locations
    registered_results_dirs = {
        "docker": "/usr/app/results",
    }

    # All registered agent types
    registered_agents = {
        "sac": SAC,
        "td3": TD3,
        "no_action": No_Action,
        "dreamer_v3": Dreamer_V3,
        "dreamer_v3_mle_reg": Dreamer_V3_Mle_Reg,
    }

    # Shared parameters across all agents
    standard_parameters = {
        "policy": "MlpPolicy",
        "stats_window_size": 1,
    }

    def __init__(
        self,
        results_dir: str,
        experiment_name: str,
        verbose: bool = True,
    ) -> None:
        """General class used to run multiple trials"""

        # set up experiment
        self.experiment_name = experiment_name
        self.results_dir = self.registered_results_dirs[results_dir]
        self.experiment_dir = f"{self.results_dir}/{self.experiment_name}"
        self.verbose = verbose

        # Create experiment results directory
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Display experiment setup
        if self.verbose:
            device = th.device("cuda" if th.cuda.is_available() else "cpu")
            fancy_print(self.experiment_name)
            underline_print("Setup")
            print(f"Results dir: {self.experiment_dir}")
            print(f"Device: {device}")
            print()

    @abstractmethod
    def _get_trial_spec_list(
        self,
        agent_list: List[str],
        env_source_name: str,
        env_name_list: Union[List[str], List[Tuple[str, str]]],
        n_train_envs: int = 8,
        n_eval_envs: int = 8,
        n_seeds: int = 10,
        custom_reward: bool = False,
    ):
        raise NotImplementedError()

    @abstractmethod
    def _get_eval_class_list(self) -> List[Base_Eval]:
        raise NotImplementedError()

    def _get_series_spec(self) -> Optional[Series_Spec]:
        return None

    def _get_agent_env_params(self, agent_name: str, env_name: str) -> Dict:
        """Returns the hyperparameters used by an agent for a specified environment.
        If no hyperparameters are found the default will be returned."""

        # Get all agent parameters
        save_loc = f"src/hyperparameters/{agent_name}.yaml"
        all_params = {}
        if os.path.exists(save_loc):
            with open(save_loc, encoding="utf-8") as yaml_file:
                all_params = yaml.safe_load(yaml_file)

        # Merge standard params with environment specific params
        common_params: dict = all_params.get("common", {})
        env_params = all_params.get(env_name, {})
        env_args: dict = self.standard_parameters | common_params | env_params

        # Update policy network parameters
        net_width = env_args.pop("net_width", 256)
        net_depth = env_args.pop("net_depth", 3)
        activation_fn = str_to_act_fn(env_args.pop("act_fn", "Tanh"))

        policy_kwargs = env_args.pop("policy_kwargs", {})
        env_args["policy_kwargs"] = policy_kwargs | {
            "net_arch": [net_width] * net_depth,
            "activation_fn": activation_fn,
        }

        # Return environment parameters
        return env_args

    def _get_default_env_list(self, env_source_name: str):
        if env_source_name == "dmc":
            return [
                ("point_mass", "easy"),
                ("cartpole", "balance"),
                ("cartpole", "swingup"),
                ("walker", "stand"),
                ("walker", "walk"),
                ("walker", "run"),
                ("cheetah", "run"),
            ]
        else:
            raise NotImplementedError

    def _generate_custom_reward_fn(
        self,
        env_source_name: str,
        env_string_name: str,
    ) -> callable:
        """Generate custom reward function which provides high reward for \\
        attaining a specified fixed point.
        """
        # Load all fixed points
        save_file = "src/hyperparameters/fixed_points.yml"
        with open(save_file, encoding="utf-8") as yaml_file:
            all_fixed_points = yaml.safe_load(yaml_file)

        # Get environment specific fixed point
        full_env_name = f"{env_source_name}_{env_string_name}"
        env_fixed_point = all_fixed_points[full_env_name]
        env_fixed_point = np.array(env_fixed_point)

        # Create and return custom reward function
        def custom_reward_fn(self, physics: Physics):  # pylint: disable=W0613
            state = physics.state()
            diff = state - env_fixed_point
            mse = np.sqrt(np.mean(diff**2))
            margin = 3

            return 2 * sigmoid(-margin * mse)

        return custom_reward_fn

    def _train_or_eval_seeds(
        self,
        trial_spec: Trial_Spec,
        trial_counter: int,
        train_agent: bool = True,
        eval_plot: bool = True,
        eval_metric: bool = True,
    ) -> None:
        """Train possible many seeds of a single agent-environment pair"""

        # Check given parameters are suitable for training
        seed_iter = [None] if trial_spec.n_seeds is None else range(trial_spec.n_seeds)
        assert trial_spec.n_seeds is None or (
            isinstance(trial_spec.n_seeds, int) and trial_spec.n_seeds > 0
        ), "`n_seeds` must be a positive integer or None."

        # Display experiment setup
        if self.verbose:
            print_hbar()
            bold_print(f"Trail {trial_counter}: {trial_spec.trial_name}")
            print()
            underline_print("Setup")
            print(f"RL Model: {trial_spec.agent_spec.agent_name}")
            print(f"Env Source: {trial_spec.env_spec.env_source_name}")
            print(f"Env Name: {trial_spec.env_spec.env_name}")
            print(f"N Training Steps: {human_format(trial_spec.n_steps)}")
            print(f"N seeds: {trial_spec.n_seeds}")
            print(f"N envs: {trial_spec.env_spec.n_train_envs}")
            print()

        # train agent
        for seed_i in seed_iter:
            trial_results_dir = f"{self.experiment_dir}/{trial_spec.save_loc}"

            if seed_i is not None:
                trial_results_dir = f"{trial_results_dir}/seed={seed_i}"
                if self.verbose:
                    underline_print(f"Seed = {seed_i}")

            trainer = Trainer(
                trial_spec=trial_spec,
                results_dir=trial_results_dir,
                verbose=self.verbose,
            )
            try:
                if train_agent:
                    trainer.train(trial_spec.n_steps)

                if eval_plot or eval_metric:
                    trainer.eval(
                        eval_functions=self._get_eval_class_list(),
                        eval_plot=eval_plot,
                        eval_metric=eval_metric,
                    )
            except Exception as e:
                print(e)
                raise e
            finally:
                trainer.close()
            print()
        print()

    def collate_results(
        self,
        agent_list: List[str],
        env_source_name: str,
        env_name_list: Union[List[str], List[Tuple[str, str]]],
        n_seeds: int = 10,
        custom_reward: bool = False,
    ):
        print("Collating Results:")
        for eval_class in self._get_eval_class_list():
            collate_class = eval_class.get_collate_results_class()
            if collate_class is not None:
                print(f"-{eval_class.get_eval_name()}")
                collator = collate_class(
                    results_dir=self.experiment_dir,
                    agent_list=agent_list,
                    env_source_name=env_source_name,
                    env_list=env_name_list,
                    save_step="final",
                    n_seeds=n_seeds,
                    verbose=self.verbose,
                    series_spec=self._get_series_spec(),
                )
                collator.collate()
        print()

    def run_experiment(
        self,
        agent_list: List[str],
        env_source_name: str,
        env_name_list: Union[List[str], List[Tuple[str, str]]],
        n_train_envs: int = 8,
        n_eval_envs: int = 8,
        n_seeds: int = 10,
        custom_reward: bool = False,
        train_agent: bool = True,
        eval_plot: bool = True,
        eval_metric: bool = False,
        collate_results: bool = False,
    ):
        if env_name_list is None:
            env_name_list = self._get_default_env_list(env_source_name)

        trial_spec_list = self._get_trial_spec_list(
            agent_list=agent_list,
            env_source_name=env_source_name,
            env_name_list=env_name_list,
            n_train_envs=n_train_envs,
            n_eval_envs=n_eval_envs,
            n_seeds=n_seeds,
            custom_reward=custom_reward,
        )

        # Run Training
        if train_agent:
            for trial_i, trial_spec in enumerate(trial_spec_list):
                self._train_or_eval_seeds(
                    trial_spec=trial_spec,
                    trial_counter=trial_i,
                    train_agent=train_agent,
                    eval_plot=False,
                    eval_metric=False,
                )

        # Run Evaluation
        if eval_metric or eval_plot:
            for trial_i, trial_spec in enumerate(trial_spec_list):
                self._train_or_eval_seeds(
                    trial_spec=trial_spec,
                    trial_counter=trial_i,
                    train_agent=False,
                    eval_plot=eval_plot,
                    eval_metric=eval_metric,
                )

        # Collate Results
        if collate_results:
            self.collate_results(
                agent_list=agent_list,
                env_source_name=env_source_name,
                env_name_list=env_name_list,
                custom_reward=custom_reward,
                n_seeds=n_seeds,
            )

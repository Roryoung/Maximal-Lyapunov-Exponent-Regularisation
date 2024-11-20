"""This module outlines the trial class which is used to run a single trial"""

import os
import gc
import shutil
import json
from typing import Union, Tuple, List, Callable
from types import MethodType
import copy
import dataclasses

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)
from dm_control import suite
from dm_control.rl import control
import gymnasium as gym

from src.evaluation.base_eval import Base_Eval
from src.callbacks.logger import Logger
from src.callbacks.save_model import Save_Model
from src.common.wrappers import DMC_Wrapper, Gym_Wrapper
from src.common.utils import dict_to_json_dict, machine_format
from src.common.custom_types import Env_Spec, Agent_Spec, Trial_Spec


class Trainer:
    """General class used to train a single agent-environment pair"""

    def __init__(
        self,
        trial_spec: Trial_Spec,
        results_dir: str,
        verbose: bool = True,
    ):
        """General class used to train a single agent-environment pair"""
        # Set up training info
        self.results_dir = results_dir
        self.verbose = verbose

        # Set up environment
        self.env_spec = trial_spec.env_spec
        self._generate_train_and_eval_envs(self.env_spec)

        # Set up agent
        self.agent_spec = self._clean_agent_spec(trial_spec.agent_spec)
        self._update_env(f"{self.results_dir}/ckpt/final/env.pkl")
        self._generate_agent(f"{self.results_dir}/ckpt/final/agent")

        # Set up experiment
        self._set_up_dir()
        self._save_setup(trial_spec)

    def _set_up_dir(self) -> None:
        """Set up directory to save results in"""

        # Delete existing experiment if required
        if os.path.exists(self.results_dir) and not self.agent_spec.load_agent:
            shutil.rmtree(self.results_dir)

        # Create results directory
        sub_dir_list = ["eval", "train_eval", "log", "ckpt"]
        for sub_dir in sub_dir_list:
            os.makedirs(f"{self.results_dir}/{sub_dir}", exist_ok=True)

    def _clean_agent_spec(self, agent_spec: Agent_Spec) -> Agent_Spec:
        """Clean the spec before creating agent and environment"""

        agent_spec = copy.deepcopy(agent_spec)
        action_noise = agent_spec.agent_kwargs.pop("action_noise", 0)
        if isinstance(action_noise, float) or isinstance(action_noise, int):
            if action_noise > 0:
                action_space = self.env.action_space.shape[0]
                sigma = action_noise * np.ones(action_space)
                agent_spec.agent_kwargs["action_noise"] = NormalActionNoise(
                    mean=np.zeros(action_space), sigma=sigma
                )
        else:
            raise ValueError(f"{action_noise} is not a valid noise value.")

        if "learning_rate" not in agent_spec.agent_kwargs:
            agent_spec.agent_kwargs["learning_rate"] = 1e-4

        return agent_spec

    def _generate_agent(self, save_loc: str, force_load: bool = False) -> None:
        """Generate/ Load agent for training"""

        tb_log = f"{self.results_dir}/log/"
        if os.path.isfile(f"{save_loc}.zip") and (
            self.agent_spec.load_agent or force_load
        ):
            self.agent = self.agent_spec.agent_cls.load(path=save_loc, env=self.env)
            self.agent.tensorboard_log = tb_log
        else:
            self.agent = self.agent_spec.agent_cls(
                env=self.env,
                tensorboard_log=tb_log,
                **self.agent_spec.agent_kwargs,
            )

    def _generate_train_and_eval_envs(self, env_spec: Env_Spec) -> None:
        """Generate training end evaluation environments"""

        # Create training end eval copies of env args
        train_env_kwargs = copy.deepcopy(env_spec.env_kwargs or {})
        eval_env_kwargs = copy.deepcopy(env_spec.env_kwargs or {})
        eval_env_kwargs.pop("custom_reward", None)

        # Make training environment
        self.env = make_vec_env(
            env_spec.env_source_name,
            env_spec.env_name,
            env_spec.n_train_envs,
            task_kwargs=env_spec.task_kwargs,
            **train_env_kwargs,
        )

        # Make eval environment
        self.eval_env = make_vec_env(
            env_spec.env_source_name,
            env_spec.env_name,
            env_spec.n_eval_envs,
            task_kwargs=env_spec.task_kwargs,
            learn_normalize=False,
            **eval_env_kwargs,
        )

        # Update eval environment norm args if normalizing input
        if isinstance(self.env, VecNormalize):
            if hasattr(self.env, "obs_rms"):
                self.eval_env.obs_rms = self.env.obs_rms
            if hasattr(self.env, "ret_rms"):
                self.eval_env.ret_rms = self.env.ret_rms

    def _update_env(self, save_loc: str) -> None:
        # reset environments
        self.env.reset()
        self.eval_env.reset()

        # update VecNormalise envs
        if os.path.isfile(save_loc):
            print("Loading env")
            if isinstance(self.env, VecNormalize):
                self.env = VecNormalize.load(save_loc, self.env.venv)
            elif isinstance(self.env, VecEnv):
                self.env = VecNormalize.load(save_loc, self.env)
                self.eval_env = VecNormalize.load(save_loc, self.eval_env)

            if hasattr(self.env, "obs_rms"):
                self.eval_env.obs_rms = self.env.obs_rms

            if hasattr(self.env, "ret_rms"):
                self.eval_env.ret_rms = self.env.ret_rms

    def _save_setup(self, trial_spec: Trial_Spec):
        """Save trial spec to json"""
        save_file = f"{self.results_dir}/ckpt/trial_spec.json"
        with open(save_file, "w+", encoding="utf-8") as json_file:
            save_data = dict_to_json_dict(dataclasses.asdict(trial_spec))
            json.dump(save_data, json_file, indent=4)

    def train(self, n_steps: int):
        """Train a single agent for a specified environment"""

        if self.verbose:
            print("Training:")

        callbacks = [Logger(self.verbose)]
        callbacks.append(
            EvalCallback(
                self.eval_env,
                callback_after_eval=Save_Model(
                    self.results_dir,
                    self.agent_spec.save_steps,
                ),
                eval_freq=max(self.agent_spec.n_eval_steps // self.env.num_envs, 1),
                n_eval_episodes=8,
                verbose=0,
            )
        )

        n_steps = np.maximum(0, n_steps - self.agent.num_timesteps)
        self.env.reset()
        self.eval_env.reset()
        self.agent.learn(
            total_timesteps=n_steps,
            callback=callbacks,
            tb_log_name="",
            reset_num_timesteps=False,
            log_interval=1,
        )

    def eval(
        self,
        eval_functions: List[Base_Eval],
        eval_plot: bool = True,
        eval_metric: bool = True,
        eval_final: bool = True,
    ) -> None:
        """Evaluate a single agent for a specified environment"""
        if self.verbose:
            print("Evaluating:")

        # Reset environments
        self.env.reset()
        self.eval_env.reset()

        all_int_save_loc = f"{self.results_dir}/ckpt/int"
        if os.path.exists(all_int_save_loc):
            # Get list of all saved models
            all_int_save_names = os.listdir(all_int_save_loc)
            int_save_steps = [machine_format(str_num) for str_num in all_int_save_names]
            sorted_all_int_save_names = np.array(all_int_save_names)[
                np.argsort(int_save_steps).astype(int)
            ].tolist()

            if eval_final:
                sorted_all_int_save_names = [sorted_all_int_save_names[-1]]

            # Iterate over all saved models
            for int_save_name in sorted_all_int_save_names:
                # update agent and env
                self._generate_agent(
                    f"{all_int_save_loc}/{int_save_name}/agent", force_load=True
                )
                self._update_env(f"{all_int_save_loc}/{int_save_name}/env.pkl")
                print(f"-{int_save_name}")

                # eval agent
                for callback_class in eval_functions:
                    callback: Base_Eval = callback_class(
                        model=self.agent,
                        eval_env=self.eval_env,
                        env_source_name=self.env_spec.env_source_name,
                        env_name=self.env_spec.env_name,
                        results_dir=self.results_dir,
                        verbose=self.verbose,
                    )
                    callback.eval(
                        save_loc=f"{self.results_dir}/train_eval/{int_save_name}",
                        eval_plot=eval_plot,
                        eval_metric=eval_metric,
                    )
                    print()

    def close(self):
        """Clean up trail"""

        self.env.close()
        self.eval_env.close()
        if hasattr(self.agent, "_logger"):
            self.agent.logger.close()
        del self.env
        del self.eval_env
        del self.agent
        th.cuda.empty_cache()
        gc.collect()


def make_vec_env(
    env_source_name: str,
    env_name: Union[str, Tuple[str, str]],
    n_envs: int,
    custom_reward: Callable = None,
    task_kwargs: dict = None,
    environment_kwargs: dict = None,
    norm_obs: bool = True,
    norm_reward: bool = True,
    learn_normalize: bool = True,
    gamma: float = 0.99,
    clip_obs: float = 10.0,
    obs_noise: float = 0,
    action_noise: float = 0,
    time_limit: Union[float, None] = None,
    n_skip_frames: int = 0,
) -> Union[DummyVecEnv, SubprocVecEnv]:
    """Function to create OpenAI Gym environment with the option of
    running multiple environments in parallel."""

    environment_kwargs = environment_kwargs or {}
    task_kwargs = task_kwargs or {}

    def make_env():
        if env_source_name == "dmc":
            assert isinstance(env_name, tuple) and len(env_name) == 2

            if time_limit is not None:
                task_kwargs["time_limit"] = time_limit

            env: control.Environment = suite.load(
                domain_name=env_name[0],
                task_name=env_name[1],
                task_kwargs=task_kwargs,
                environment_kwargs={"flat_observation": True} | environment_kwargs,
            )
            env = DMC_Wrapper(
                env,
                obs_noise=obs_noise,
                action_noise=action_noise,
                n_skip_frames=n_skip_frames,
            )

            if custom_reward:
                env.task.get_reward = MethodType(custom_reward, env.task)

        elif env_source_name == "gym":
            env = gym.make(
                env_name,
                render_mode="rgb_array",
            )
            env = Gym_Wrapper(env)
        else:
            raise ValueError(f"Env source name {env_source_name} is not recognised.")

        check_env(env)
        return Monitor(env, None)

    if n_envs is None or n_envs <= 1:
        vec_env = DummyVecEnv([make_env])
    else:
        vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])

    if not norm_obs and not norm_reward:
        return vec_env

    return VecNormalize(
        vec_env,
        training=learn_normalize,
        norm_obs=norm_obs,
        norm_reward=norm_reward and learn_normalize,
        gamma=gamma,
        clip_obs=clip_obs,
    )

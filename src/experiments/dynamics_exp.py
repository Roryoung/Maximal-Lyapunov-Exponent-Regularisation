from typing import Union, List, Tuple
import argparse
import copy

from src.common.experimenter import Base_Experimenter, env_name_type
from src.common.custom_types import Env_Spec, Agent_Spec, Trial_Spec
from src.evaluation import *


class Dynamics_Experiment(Base_Experimenter):
    def _get_eval_class_list(self):
        return [
            Eval_Reward,
            Render,
            Eval_Determinism,
            Plt_MLE_Spectrum,
            Observation_Attack,
        ]

    def _get_trial_spec_list(
        self,
        agent_list: List[str],
        env_source_name: str,
        env_name_list: Union[List[str], List[Tuple[str, str]]],
        n_train_envs: int = 8,
        n_eval_envs: int = 8,
        n_seeds: int = 10,
        custom_reward: bool = False,
    ) -> List[Trial_Spec]:
        trial_spec_list = []
        for agent in agent_list:
            for env_name in env_name_list:
                if isinstance(env_name, tuple):
                    env_string_name = "_".join(env_name)
                else:
                    env_string_name = env_name

                # Get agent and environment specific parameters
                task_params = self._get_agent_env_params(agent, env_string_name)
                total_timesteps = task_params.pop("total_timesteps", 1_000_000)

                # Split all specified parameters into agent and environment ones
                env_kwargs = task_params.get("env_kwargs", {})

                if custom_reward:
                    env_kwargs["custom_reward"] = self._generate_custom_reward_fn(
                        env_source_name, env_string_name
                    )
                    env_string_name = f"fp_{env_string_name}"

                env_kwargs["norm_obs"] = task_params.pop("norm_obs", False)
                env_kwargs["norm_reward"] = task_params.pop("norm_reward", False)
                env_kwargs["gamma"] = task_params.pop("gamma", 0.99)

                # Create Trial Spec
                env_spec = Env_Spec(
                    env_source_name=env_source_name,
                    env_name=env_name,
                    task_kwargs={},
                    env_kwargs=env_kwargs,
                    n_train_envs=n_train_envs,
                    n_eval_envs=n_eval_envs,
                )

                agent_spec = Agent_Spec(
                    agent_name=agent,
                    agent_cls=self.registered_agents[agent],
                    agent_kwargs=copy.deepcopy(task_params),
                    n_eval_steps=100_000,
                    save_steps=1,
                )

                trial_name = f"{env_source_name}_{env_string_name}"
                trial_spec_list.append(
                    Trial_Spec(
                        trial_name=trial_name,
                        env_spec=env_spec,
                        agent_spec=agent_spec,
                        save_loc=f"{agent}/{trial_name}",
                        n_steps=total_timesteps,
                        n_seeds=n_seeds,
                    )
                )

        return trial_spec_list


def run_experiment(
    agent: List[str],
    results_dir: str,
    env_source_name: str,
    env_name: Union[List[str], List[Tuple[str, str]]],
    train_agent: bool = True,
    eval_agent: bool = False,
    eval_plot_agent: bool = False,
    collate_results: bool = False,
    custom_reward: bool = False,
    n_envs: int = 8,
    n_eval_envs: int = 8,
    n_seeds: int = 10,
):
    experiment = Dynamics_Experiment(
        results_dir=results_dir,
        experiment_name="dynamics_experiment",
    )

    experiment.run_experiment(
        agent_list=agent,
        env_source_name=env_source_name,
        env_name_list=env_name,
        n_train_envs=n_envs,
        n_eval_envs=n_eval_envs,
        n_seeds=n_seeds,
        custom_reward=custom_reward,
        train_agent=train_agent,
        eval_plot=eval_plot_agent,
        eval_metric=eval_agent,
        collate_results=collate_results,
    )


def main():
    # Collect command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", required=True, nargs="+")
    parser.add_argument("-s", "--env_source_name", type=str, required=True)
    parser.add_argument("-e", "--env_name", type=env_name_type, nargs="+")
    parser.add_argument("-r", "--results_dir", default="docker", choices=["docker"])
    parser.add_argument("--train", action="store_true", dest="train_agent")
    parser.add_argument("--eval", action="store_true", dest="eval_agent")
    parser.add_argument("--eval_plot", action="store_true", dest="eval_plot_agent")
    parser.add_argument("--collate_results", action="store_true")
    parser.add_argument("--custom_reward", action="store_true")
    parser.add_argument("--n_envs", default=8, type=int)
    parser.add_argument("--n_eval_envs", default=8, type=int)
    parser.add_argument("--n_seeds", default=10, type=int)
    args = parser.parse_args()

    # Run experiment
    run_experiment(**vars(args))


if __name__ == "__main__":
    main()

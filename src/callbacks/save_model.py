import os

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from src.common.utils import human_format


class Save_Model(BaseCallback):
    def __init__(self, results_dir, save_steps: int, verbose=0):
        super(Save_Model, self).__init__(verbose)
        self.results_dir = f"{results_dir}/ckpt"
        self.int_results_dir = f"{results_dir}/ckpt/int"
        self.save_steps = save_steps
        os.makedirs(self.results_dir, exist_ok=True)

    def _save_model(self, save_loc: str) -> None:
        os.makedirs(save_loc, exist_ok=True)

        # Save model
        self.model.save(f"{save_loc}/agent.zip")

        # Save normalized env
        if isinstance(self.model.env, VecNormalize):
            self.model.env.save(f"{save_loc}/env.pkl")

    def _on_training_start(self) -> None:
        n_steps = human_format(self.model.num_timesteps)
        self._save_model(f"{self.int_results_dir}/{n_steps}")
        return True

    def _on_step(self) -> bool:
        if self.n_calls % self.save_steps == 0:
            n_steps = human_format(self.model.num_timesteps)
            self._save_model(f"{self.int_results_dir}/{n_steps}")
            self._save_model(f"{self.results_dir}/final")
            print()

        return True

    def _on_training_end(self) -> None:
        self._save_model(f"{self.results_dir}/final")
        return True

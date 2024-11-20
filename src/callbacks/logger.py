import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.common.utils import print_line


class Logger(BaseCallback):
    def __init__(self, verbose: bool = True, eval_end: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.eval_end = eval_end

    def _on_step(self) -> bool:
        n_interactions = self.model.num_timesteps
        total_interactions = self.locals["total_timesteps"]
        eval_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
        mean_reward = 0.0 if len(eval_rewards) == 0 else np.mean(eval_rewards)

        print_line(
            f"[{n_interactions:.0f}]/[{total_interactions:.0f}] | mean_reward = {mean_reward:.4f}"
        )

        return super()._on_step()

    def _on_training_end(self) -> None:
        if self.verbose and self.eval_end:
            print("\n")
            print("Evaluating:")
        return super()._on_training_end()

    def eval(self, save_loc: str, n_inits: int) -> None:
        pass

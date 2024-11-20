"""
Replay buffer used for Dreamer V3 and Dreamer V3 + MLE regularisation.
This work is based on the Electric Sheep implementation of Dreamer V3 [1]

[1]: https://github.com/Eclectic-Sheep/sheeprl/tree/main/sheeprl/algos/dreamer_v3
"""

from typing import Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer


class Sequential_Replay_Buffer_Samples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    is_first: th.Tensor


class Sequential_Replay_Buffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
        self.dones[-1] = True

    def sample(
        self,
        sequence_length: int,
        batch_size: int,
    ) -> Sequential_Replay_Buffer_Samples:
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError(
                f"'batch_size' ({batch_size}) and 'n_samples' ({sequence_length}) must be both greater than 0"
            )
        if not self.full and self.pos == 0:
            raise ValueError(
                "No sample has been added to the buffer. Please add at least one sample calling 'self.add()'"
            )
        if not self.full and self.pos - sequence_length + 1 < 1:
            raise ValueError(
                f"Cannot sample a sequence of length {sequence_length}. Data added so far: {self.pos}"
            )
        if self.full and sequence_length > self.buffer_size:
            raise ValueError(
                f"The sequence length ({sequence_length}) is greater than the buffer size ({self.buffer_size})"
            )

        # get sequence inds
        if self.full:
            first_range_end = self.pos - sequence_length + 1
            second_range_end = (
                self.buffer_size
                if first_range_end >= 0
                else self.buffer_size + first_range_end
            )

            valid_inds = np.array(
                list(range(0, first_range_end))
                + list(range(self.pos, second_range_end)),
                dtype=np.intp,
            )

            start_inds = valid_inds[
                np.random.randint(
                    low=0,
                    high=len(valid_inds),
                    size=batch_size,
                )
            ]
        else:
            start_inds = np.random.randint(
                low=0,
                high=self.pos - sequence_length + 1,
                size=batch_size,
            )

        chunk_length = np.arange(sequence_length, dtype=np.intp).reshape(-1, 1)
        seq_inds = (start_inds.reshape(1, -1) + chunk_length) % self.buffer_size

        # get batch inds
        batch_inds = np.random.randint(0, self.n_envs, batch_size)
        batch_inds = batch_inds[None, :].repeat(sequence_length, 0)

        return self._get_samples(seq_inds, batch_inds)

    def _get_samples(
        self,
        seq_inds: np.ndarray,
        batch_inds: np.ndarray,
    ) -> Sequential_Replay_Buffer_Samples:
        return Sequential_Replay_Buffer_Samples(
            observations=self.to_torch(self.observations[seq_inds, batch_inds]).float(),
            actions=self.to_torch(self.actions[seq_inds, batch_inds]).float(),
            rewards=self.to_torch(self.rewards[seq_inds, batch_inds, None]).float(),
            dones=self.to_torch(self.dones[seq_inds, batch_inds, None]).float(),
            is_first=self.to_torch(self.dones[seq_inds - 1, batch_inds, None]).float(),
        )

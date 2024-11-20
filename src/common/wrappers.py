import copy
from typing import Any, SupportsFloat, Union

from gymnasium import Env, spaces
from dm_env import specs
import numpy as np
from dm_control.rl.control import Environment, flatten_observation


def _spec_to_box(spec, dtype=np.float64):
    shape = spec.shape
    if type(spec) == specs.Array or isinstance(spec, np.ndarray):
        high = np.inf * np.ones(shape, dtype=dtype)
        low = -high

    elif type(spec) == specs.BoundedArray:
        high = spec.maximum.astype(dtype)
        low = spec.minimum.astype(dtype)

    return spaces.Box(low, high, shape=shape, dtype=dtype)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs:
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMC_Wrapper(Env):
    def __init__(
        self,
        dmc_env: Environment,
        obs_noise: float = 0,
        action_noise: float = 0,
        seed=None,
        height=480,
        width=640,
        camera_id=[0],
        n_skip_frames=0,
        render_reward=False,
        render_mode="rgb_array",
    ):
        # set up env info
        self.dmc_env = dmc_env
        self._obs_noise_std = obs_noise
        self._action_noise_std = action_noise
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._n_skip_frames = n_skip_frames
        self.render_mode = render_mode
        self.render_reward = render_reward
        self.ep_reward = 0
        self.ep_len = 0

        # set up spaces
        self._action_space = _spec_to_box(self.dmc_env.action_spec(), dtype=np.float32)
        self._observation_space = _spec_to_box(
            self.dmc_env.observation_spec()["observations"]
        )

        # reset env
        self.current_time_step = dmc_env.reset()

        # set seed
        if seed is not None:
            self.seed(seed=seed)

    def __getattr__(self, name):
        return getattr(self.dmc_env, name)

    def _get_obs(self, time_step):
        obs = _flatten_obs(time_step.observation["observations"])
        noise = np.random.normal(0, self._obs_noise_std, size=obs.shape)
        return obs + noise

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def reward_range(self):
        return 0, 1

    def seed(self, seed):
        self._action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        action = np.array(action)
        reward = 0
        info = {}

        self.get_physics()._data.ptr.qacc_warmstart[:] = 0
        for _ in range(self._n_skip_frames + 1):
            time_step = self.dmc_env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()

            if done:
                break

        obs = self._get_obs(time_step)
        self.current_time_step = time_step
        self.ep_reward += reward
        self.ep_len += 1
        info["discount"] = time_step.discount
        info["is_success"] = done
        if done == True:
            info["episode"] = {
                "r": self.ep_reward,
                "l": self.ep_len,
            }

            self.ep_reward = 0
            self.ep_len = 0

        return obs, reward, False, done, info

    def reset(self, seed=None, options=None):
        self.get_physics()._data.ptr.qacc_warmstart[:] = 0
        self.current_time_step = self.dmc_env.reset()
        obs = self._get_obs(self.current_time_step)
        self.ep_reward = 0
        self.ep_len = 0
        return obs, {}

    def set_state(self, state):
        self.dmc_env._physics.reset()
        self.dmc_env._physics.set_state(state)

        with self.dmc_env._physics.model.disable("actuation"):
            self.dmc_env._physics.forward()

        obs = self.dmc_env._task.get_observation(self.dmc_env._physics)
        if self.dmc_env._flat_observation:
            obs = flatten_observation(obs)
            obs = _flatten_obs(obs["observations"])
            obs += np.random.normal(0, self._obs_noise_std, size=obs.shape)

        return obs

    def get_state(self):
        return self.dmc_env._physics.get_state()

    def set_physics(self, physics):
        self.dmc_env._physics = physics

    def get_physics(self):
        return self.dmc_env._physics

    def set_task(self, task):
        self.dmc_env._task = copy.deepcopy(task)

    def get_task(self):
        return copy.deepcopy(self.dmc_env._task)

    def get_dmc_env(self):
        return self.dmc_env

    def render(self):
        self._set_reward_colors(self.current_time_step.reward or 0)

        images = [
            self.dmc_env.physics.render(
                height=self._height, width=self._width, camera_id=camera_id
            )
            for camera_id in self._camera_id
        ]

        return np.concatenate(images, axis=0)

    def render_angles(self, camera_ids=None):
        self._camera_id = camera_ids

    def _set_reward_colors(self, reward):
        """Sets the highlight, effector and target colors according to the reward."""
        _MATERIALS = ["self", "effector", "target"]
        _DEFAULT = [name + "_default" for name in _MATERIALS]
        _HIGHLIGHT = [name + "_highlight" for name in _MATERIALS]

        if self.render_reward:
            assert 0.0 <= reward <= 1.0
            colors = self.dmc_env.physics.named.model.mat_rgba
            default = colors[_DEFAULT]
            highlight = colors[_HIGHLIGHT]
            blend_coef = reward**4
            colors[_MATERIALS] = blend_coef * highlight + (1.0 - blend_coef) * default


class Gym_Wrapper(Env):
    def __init__(
        self,
        gym_env: Env,
        obs_noise: float = 0,
        action_noise: float = 0,
        seed=None,
        n_skip_frames=0,
    ) -> None:
        self.gym_env = gym_env
        self.n_skip_frames = n_skip_frames
        self.qpos_len, self.qvel_len = [len(q_val) for q_val in self._get_q_vals()]
        self.render_mode = self.gym_env.render_mode

        if seed is not None:
            self.seed(seed=seed)

    def __getattr__(self, name):
        return getattr(self.gym_env, name)

    def seed(self, seed):
        self.gym_env.action_space.seed(seed)
        self.gym_env.observation_space.seed(seed)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        total_reward = 0
        for i in range(self.n_skip_frames + 1):
            obs, reward, term, trunc, info = self.gym_env.step(action)
            total_reward += reward or 0

            if term or trunc:
                break

        return obs, total_reward, term, trunc, info

    def reset(
        self, seed: Union[int, None] = None, options: Union[dict[str, Any], None] = None
    ) -> tuple[Any, dict[str, Any]]:
        return self.gym_env.reset()

    def render(self):
        return self.gym_env.render()

    def _get_q_vals(self):
        qpos = np.copy(self.gym_env.unwrapped.data.qpos[:])
        qvel = np.copy(self.gym_env.unwrapped.data.qvel[:])
        return qpos, qvel

    def get_state(self):
        return np.concatenate(self._get_q_vals())

    def set_state(self, state):
        assert len(state) == (self.qpos_len + self.qvel_len)
        self.reset()

        qpos = state[: self.qpos_len]
        qvel = state[self.qpos_len :]

        self.gym_env.unwrapped.set_state(qpos, qvel)
        return self.gym_env.unwrapped._get_obs()

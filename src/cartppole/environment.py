from typing import Final, NamedTuple, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from cartppole.typing import Shaped

# https://gymnasium.farama.org/environments/classic_control/cart_pole/
#
# cartpole has
# - 4 observation dims, and
# - 2 discrete actions
ENVIRONMENT_ID: Final[str] = "CartPole-v1"
OBS_DIM: Final[int] = 4
ACT_DIM: Final[int] = 2


class Reset(NamedTuple):
    obs: Shaped[npt.NDArray[np.float32], "n_envs 4"]
    info: dict


class Step(NamedTuple):
    obs: Shaped[npt.NDArray[np.float32], "n_envs 4"]
    reward: Shaped[npt.NDArray[np.float32], "n_envs"]
    terminated: Shaped[npt.NDArray[np.bool], "n_envs"]
    truncated: Shaped[npt.NDArray[np.bool], "n_envs"]
    info: dict


class Environment:
    n_envs: int

    _vec: gym.vector.VectorEnv
    _scalar: gym.Env

    def __init__(self, n_envs: int = 1, render: bool = False) -> None:
        self.n_envs = n_envs
        self.render = render
        if n_envs > 1:
            assert not render, "rendering only supported for n_envs=1"
            self._vec = gym.make_vec(id=ENVIRONMENT_ID, num_envs=n_envs)
        else:
            self._scalar = gym.make(
                id=ENVIRONMENT_ID,
                render_mode="human" if render else None,
            )

    @property
    def observation_space(self) -> gym.spaces.Box:
        if self.n_envs > 1:
            return cast(gym.spaces.Box, self._vec.single_observation_space)
        else:
            return cast(gym.spaces.Box, self._scalar.observation_space)

    @property
    def action_space(self) -> gym.spaces.Discrete:
        if self.n_envs > 1:
            return cast(gym.spaces.Discrete, self._vec.single_action_space)
        else:
            return cast(gym.spaces.Discrete, self._scalar.action_space)

    def reset(self, seed: int | None = None) -> Reset:
        if self.n_envs > 1:
            obs, info = self._vec.reset(seed=seed)
            return Reset(obs=obs, info=info)
        else:
            obs, info = self._scalar.reset(seed=seed)
            return Reset(
                obs=np.expand_dims(obs, axis=0),
                info=info,
            )

    def step(self, actions: Shaped[npt.NDArray[np.int64], "n_envs"]) -> Step:
        if self.n_envs > 1:
            obs, reward, terminated, truncated, info = self._vec.step(actions)
            return Step(
                obs=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        else:
            assert actions.size == 1
            obs, reward, terminated, truncated, info = self._scalar.step(
                int(actions.item())
            )
            return Step(
                obs=np.expand_dims(obs, axis=0).astype(np.float32),
                reward=np.array([reward], dtype=np.float32),
                terminated=np.array([terminated], dtype=np.bool_),
                truncated=np.array([truncated], dtype=np.bool_),
                info=info,
            )

    def close(self) -> None:
        if self.n_envs > 1:
            self._vec.close()
        else:
            self._scalar.close()

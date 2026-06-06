from enum import StrEnum
from typing import Annotated, Final, NamedTuple, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt


class EnvironmentID(StrEnum):
    CartPoleV1 = "CartPole-v1"
    AcrobotV1 = "Acrobot-v1"
    MountainCarV0 = "MountainCar-v0"


class Reset(NamedTuple):
    obs: Annotated[npt.NDArray[np.float32], "n_envs obs_dim"]
    info: dict


class Step(NamedTuple):
    obs: Annotated[npt.NDArray[np.float32], "n_envs obs_dim"]
    reward: Annotated[npt.NDArray[np.float32], "n_envs"]
    terminated: Annotated[npt.NDArray[np.bool_], "n_envs"]
    truncated: Annotated[npt.NDArray[np.bool_], "n_envs"]
    info: dict

    @property
    def done(self) -> Annotated[npt.NDArray[np.bool_], "n_envs"]:
        return self.terminated | self.truncated


class Environment:
    n_envs: int

    _vec: gym.vector.VectorEnv
    _scalar: gym.Env

    def __init__(
        self,
        id: str | EnvironmentID = EnvironmentID.CartPoleV1,
        n_envs: int = 1,
        render: bool = False,
    ) -> None:
        self.n_envs = n_envs
        self.render = render
        if n_envs > 1:
            assert not render, "rendering only supported for n_envs=1"
            self._vec = gym.make_vec(id=id, num_envs=n_envs)
        else:
            self._scalar = gym.make(
                id=id,
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

    @property
    def obs_dim(self) -> int:
        return int(self.observation_space.shape[0])

    @property
    def act_dim(self) -> int:
        return int(self.action_space.n)

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

    def step(self, actions: Annotated[npt.NDArray[np.int64], "n_envs"]) -> Step:
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
                terminated=np.array([terminated], dtype=np.bool),
                truncated=np.array([truncated], dtype=np.bool),
                info=info,
            )

    def close(self) -> None:
        if self.n_envs > 1:
            self._vec.close()
        else:
            self._scalar.close()

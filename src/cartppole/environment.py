from enum import StrEnum
from typing import Annotated, NamedTuple, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt


class EnvironmentID(StrEnum):
    """Supported Gymnasium environment identifiers."""

    CartPoleV1 = "CartPole-v1"
    AcrobotV1 = "Acrobot-v1"
    MountainCarV0 = "MountainCar-v0"


class Reset(NamedTuple):
    """Batched result returned by ``Environment.reset``.

    ``obs`` always has a leading environment dimension, even when wrapping a
    single scalar Gymnasium environment.
    """

    obs: Annotated[npt.NDArray[np.float32], "n_envs obs_dim"]
    info: dict


class Step(NamedTuple):
    """Batched transition returned by ``Environment.step``.

    The wrapper keeps the Gymnasium ``terminated`` and ``truncated`` flags
    separate while also exposing a convenience ``done`` property.
    """

    obs: Annotated[npt.NDArray[np.float32], "n_envs obs_dim"]
    reward: Annotated[npt.NDArray[np.float32], "n_envs"]
    terminated: Annotated[npt.NDArray[np.bool_], "n_envs"]
    truncated: Annotated[npt.NDArray[np.bool_], "n_envs"]
    info: dict

    @property
    def done(self) -> Annotated[npt.NDArray[np.bool_], "n_envs"]:
        """Return terminal-or-truncated flags for each environment."""
        return self.terminated | self.truncated


class Environment:
    """Uniform wrapper around scalar and vector Gymnasium environments.

    Training uses vector environments for faster rollout collection, while
    rendering and evaluation use a single scalar environment. This wrapper
    exposes both modes through one small batched API.
    """

    n_envs: int

    _vec: gym.vector.VectorEnv
    _scalar: gym.Env

    def __init__(
        self,
        id: str | EnvironmentID = EnvironmentID.CartPoleV1,
        n_envs: int = 1,
        render: bool = False,
    ) -> None:
        """Create a CartPole-compatible environment wrapper.

        Args:
            id: Gymnasium environment identifier.
            n_envs: Number of parallel environments to create.
            render: Whether to enable human rendering for a single environment.

        Raises:
            AssertionError: If rendering is requested with ``n_envs > 1``.
        """
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
        """Return the single-environment observation space."""
        if self.n_envs > 1:
            return cast(gym.spaces.Box, self._vec.single_observation_space)
        else:
            return cast(gym.spaces.Box, self._scalar.observation_space)

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """Return the single-environment discrete action space."""
        if self.n_envs > 1:
            return cast(gym.spaces.Discrete, self._vec.single_action_space)
        else:
            return cast(gym.spaces.Discrete, self._scalar.action_space)

    @property
    def obs_dim(self) -> int:
        """Return the flattened observation dimensionality."""
        return int(self.observation_space.shape[0])

    @property
    def act_dim(self) -> int:
        """Return the number of discrete actions."""
        return int(self.action_space.n)

    def reset(self, seed: int | None = None) -> Reset:
        """Reset the environment and return batched observations.

        Args:
            seed: Optional Gymnasium reset seed.

        Returns:
            Batched observations and Gymnasium reset info.
        """
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
        """Apply a batch of actions and return a batched transition.

        Args:
            actions: One discrete action per environment.

        Returns:
            Batched observations, rewards, termination flags, truncation flags,
            and Gymnasium info.
        """
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
        """Release the underlying Gymnasium environment resources."""
        if self.n_envs > 1:
            self._vec.close()
        else:
            self._scalar.close()

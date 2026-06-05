from typing import Annotated, NamedTuple

from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Linear, Module, Sequential, Tanh


class PolicyOutput(NamedTuple):
    action: Annotated[Tensor, "batch_size"]
    log_prob: Annotated[Tensor, "batch_size"]
    entropy: Annotated[Tensor, "batch_size"]
    value: Annotated[Tensor, "batch_size"]


class Policy(Module):
    """Actor-critic policy network for PPO on discrete-action environments.

    The actor maps each observation to action logits. These logits parameterize a
    categorical distribution over the discrete actions, from which actions can be
    sampled and log-probabilities can be computed for PPO's policy loss.

    The critic maps each observation to a scalar state-value estimate ``V(s)``.
    PPO uses this value estimate to compute advantages and the value-function
    loss.

    Args:
        obs_dim: Number of observation features. CartPole uses ``4``.
        act_dim: Number of discrete actions. CartPole uses ``2``.
        hidden_dim: Width of each hidden layer in the actor and critic MLPs.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self._actor = Sequential(
            Linear(obs_dim, hidden_dim),
            Tanh(),
            Linear(hidden_dim, hidden_dim),
            Tanh(),
            Linear(hidden_dim, act_dim),
        )
        self._critic = Sequential(
            Linear(obs_dim, hidden_dim),
            Tanh(),
            Linear(hidden_dim, hidden_dim),
            Tanh(),
            Linear(hidden_dim, 1),
        )

    def get_value(
        self,
        obs: Annotated[Tensor, "batch_size obs_dim"],
    ) -> Annotated[Tensor, "batch_size"]:
        """Estimate the value of each observation.

        Args:
            obs: Batch of observations.

        Returns:
            One scalar value estimate per observation.
        """
        return self._critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: Annotated[Tensor, "batch_size obs_dim"],
        action: Annotated[Tensor, "batch_size"] | None = None,
    ) -> PolicyOutput:
        """Sample or evaluate actions and compute critic values.

        The actor produces one logit per action. These logits define a
        categorical distribution over discrete actions. If ``action`` is
        ``None``, an action is sampled from that distribution. If ``action`` is
        provided, that existing action is evaluated under the current policy.

        PPO needs both modes: sampling actions while collecting rollouts, and
        recomputing log-probabilities for stored actions during the update.

        Args:
            obs: Batch of observations.
            action: Optional batch of discrete actions.

        Returns:
            The action, its log-probability, the distribution entropy, and the
            critic value estimate.
        """
        logits: Annotated[Tensor, "batch_size act_dim"] = self._actor(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.get_value(obs)

        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )

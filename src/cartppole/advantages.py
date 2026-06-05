from typing import Annotated, NamedTuple

from torch import Tensor, zeros_like


class Advantage(NamedTuple):
    returns: Annotated[Tensor, "n_steps n_envs"]
    advantages: Annotated[Tensor, "n_steps n_envs"]


def monte_carlo(
    rewards: Annotated[Tensor, "n_steps n_envs"],
    dones: Annotated[Tensor, "n_steps n_envs"],
    values: Annotated[Tensor, "n_steps n_envs"],
    next_value: Annotated[Tensor, "n_envs"],
    next_done: Annotated[Tensor, "n_envs"],
    discount_factor: float,
) -> Advantage:
    r"""Compute Monte Carlo discounted returns and advantages.

    The return target is the discounted sum of future rewards:

    $$
    G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
    $$

    Recursively, this is:

    $$
    G_t = r_t + \gamma (1 - d_{t+1}) G_{t+1}
    $$

    where ``d`` is the done flag. If the next state is terminal, the
    ``(1 - d)`` term becomes zero, so the return does not bootstrap across
    episode boundaries.

    The advantage is then:

    $$
    A_t = G_t - V(s_t)
    $$

    Args:
        rewards: Rewards.
        dones: Done flags.
        values: Critic estimates ``V(s_t)``.
        next_value: Critic estimate for the observation after the rollout.
        next_done: Done flags for the observation after the rollout.
        discount_factor: Gamma.

    Returns:
        Discounted returns and advantages.
    """
    returns: Annotated[Tensor, "n_steps n_envs"] = zeros_like(rewards)

    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_nonterminal = 1.0 - next_done
            next_return = next_value
        else:
            next_nonterminal = 1.0 - dones[step + 1]
            next_return = returns[step + 1]

        returns[step] = rewards[step] + discount_factor * next_nonterminal * next_return

    advantages: Annotated[Tensor, "n_steps n_envs"] = returns - values
    return Advantage(returns=returns, advantages=advantages)

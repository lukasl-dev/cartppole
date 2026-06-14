from typing import Annotated, NamedTuple

from torch import Tensor, zeros_like


class Advantage(NamedTuple):
    ret: Annotated[Tensor, "n_steps n_envs"]
    adv: Annotated[Tensor, "n_steps n_envs"]


def monte_carlo(
    rew: Annotated[Tensor, "n_steps n_envs"],
    dones: Annotated[Tensor, "n_steps n_envs"],
    val: Annotated[Tensor, "n_steps n_envs"],
    next_val: Annotated[Tensor, "n_envs"],
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
    returns: Annotated[Tensor, "n_steps n_envs"] = zeros_like(rew)

    for step in reversed(range(rew.shape[0])):
        if step == rew.shape[0] - 1:
            next_nonterminal = 1.0 - next_done
            next_return = next_val
        else:
            next_nonterminal = 1.0 - dones[step + 1]
            next_return = returns[step + 1]

        returns[step] = rew[step] + discount_factor * next_nonterminal * next_return

    advantages: Annotated[Tensor, "n_steps n_envs"] = returns - val
    return Advantage(ret=returns, adv=advantages)


def generalised_advantage_estimation(
    rew: Annotated[Tensor, "n_steps n_envs"],
    dones: Annotated[Tensor, "n_steps n_envs"],
    val: Annotated[Tensor, "n_steps n_envs"],
    next_val: Annotated[Tensor, "n_envs"],
    next_done: Annotated[Tensor, "n_envs"],
    discount_factor: float,
    gae_lambda: float,
) -> Advantage:
    r"""Compute Generalised Advantage Estimation (GAE).

    GAE computes advantages by exponentially weighting multi-step temporal
    difference residuals:

    $$
    \delta_t = r_t + \gamma (1 - d_{t+1}) V(s_{t+1}) - V(s_t)
    $$

    $$
    A_t^\mathrm{GAE} = \delta_t + \gamma \lambda (1 - d_{t+1})
        A_{t+1}^\mathrm{GAE}
    $$

    ``lambda`` controls the bias/variance trade-off: lower values rely more on
    the critic's one-step TD errors, while values near ``1`` approach Monte
    Carlo-style advantages. Value targets for PPO's critic are then:

    $$
    R_t = A_t^\mathrm{GAE} + V(s_t)
    $$

    Args:
        rew: Rewards.
        dones: Done flags stored at the start of each rollout step. Therefore,
            ``dones[step + 1]`` is the terminal mask for transition ``step``.
        val: Critic estimates ``V(s_t)``.
        next_val: Critic estimate for the observation after the rollout.
        next_done: Done flags for the observation after the rollout.
        discount_factor: Gamma.
        gae_lambda: GAE lambda.

    Returns:
        Lambda-return value targets and GAE advantages.
    """
    advantages: Annotated[Tensor, "n_steps n_envs"] = zeros_like(rew)
    last_gae_lam = zeros_like(next_val)

    for step in reversed(range(rew.shape[0])):
        # Use rollout bootstrap values at the end; otherwise use the next stored step.
        if step == rew.shape[0] - 1:
            next_nonterminal = 1.0 - next_done
            next_value = next_val
        else:
            next_nonterminal = 1.0 - dones[step + 1]
            next_value = val[step + 1]

        delta = rew[step] + discount_factor * next_nonterminal * next_value - val[step]
        last_gae_lam = (
            delta + discount_factor * gae_lambda * next_nonterminal * last_gae_lam
        )
        advantages[step] = last_gae_lam

    returns: Annotated[Tensor, "n_steps n_envs"] = advantages + val
    return Advantage(ret=returns, adv=advantages)

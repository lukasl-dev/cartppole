from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
import torch
from torch import no_grad, tensor

from cartppole.environment import Environment
from cartppole.policy import Policy


class EvaluationResult(NamedTuple):
    """Episode-return statistics for a saved PPO checkpoint."""

    checkpoint_path: Path
    n_episodes: int
    success_threshold: float
    episode_returns: list[float]
    return_mean: float
    return_std: float
    return_min: float
    return_max: float
    success_rate: float


def evaluate(
    env_id: str = "CartPole-v1",
    checkpoint_path: Path = Path("checkpoints/policy.pt"),
    success_threshold: float = 475,
    n_episodes: int = 60,
    seed: int = 0,
    deterministic: bool = False,
) -> EvaluationResult:
    """Evaluate a saved policy checkpoint on fresh episodes.

    Args:
        env_id: Gymnasium environment identifier.
        checkpoint_path: Path to a checkpoint produced by ``train``.
        success_threshold: Episode return counted as a successful episode.
        n_episodes: Number of episodes to evaluate.
        seed: Base seed; episode ``i`` uses ``seed + i``.
        deterministic: Whether to choose greedy actions instead of sampling.

    Returns:
        Per-episode returns and aggregate evaluation statistics.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    env = Environment(id=env_id, n_envs=1)
    policy = Policy(
        obs_dim=checkpoint["obs_dim"],
        act_dim=checkpoint["act_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    episode_returns: list[float] = []

    try:
        for episode in range(n_episodes):
            reset = env.reset(seed=seed + episode)
            obs = tensor(reset.obs, dtype=torch.float32)
            total_reward = 0.0

            while True:
                with no_grad():
                    out = policy.get_action_and_value(obs, deterministic=deterministic)

                step = env.step(out.action.numpy())
                total_reward += float(step.reward[0])
                obs = tensor(step.obs, dtype=torch.float32)

                if bool(step.done[0]):
                    episode_returns.append(total_reward)
                    break
    finally:
        env.close()

    returns = np.array(episode_returns)
    return EvaluationResult(
        checkpoint_path=checkpoint_path,
        n_episodes=n_episodes,
        success_threshold=success_threshold,
        episode_returns=episode_returns,
        return_mean=float(np.mean(returns)),
        return_std=float(np.std(returns)),
        return_min=float(np.min(returns)),
        return_max=float(np.max(returns)),
        success_rate=float(np.mean(returns >= success_threshold)),
    )


@click.command()
@click.option(
    "--env",
    "env_id",
    default="CartPole-v1",
    show_default=True,
    type=str,
    help="Gymnasium environment ID.",
)
@click.option(
    "--checkpoint-path",
    default="checkpoints/policy.pt",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--success-threshold", default=475.0, show_default=True, type=float)
@click.option("--n-episodes", default=60, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option(
    "--deterministic/--stochastic",
    default=False,
    show_default=True,
    help="Use greedy actions during evaluation instead of sampling.",
)
def evaluate_cli(
    env_id: str = "CartPole-v1",
    checkpoint_path: Path = Path("checkpoints/policy.pt"),
    success_threshold: float = 475,
    n_episodes: int = 60,
    seed: int = 0,
    deterministic: bool = False,
) -> None:
    """CLI entry point for evaluating a saved checkpoint."""
    result = evaluate(
        env_id=env_id,
        checkpoint_path=checkpoint_path,
        success_threshold=success_threshold,
        n_episodes=n_episodes,
        seed=seed,
        deterministic=deterministic,
    )
    click.echo(f"{'checkpoint:':<24}{result.checkpoint_path}")
    click.echo(f"{'episodes:':<24}{result.n_episodes}")
    click.echo(
        f"{'return mean±std:':<24}{result.return_mean:.1f} ± {result.return_std:.1f}"
    )
    click.echo(f"{'return min:':<24}{result.return_min:.1f}")
    click.echo(f"{'return max:':<24}{result.return_max:.1f}")
    click.echo(
        f"{'success rate:':<24}{result.success_rate:.2f}  "
        f"(return ≥ {result.success_threshold})"
    )


if __name__ == "__main__":
    evaluate_cli()

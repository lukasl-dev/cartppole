from pathlib import Path

import click
import numpy as np
import torch
from torch import no_grad, tensor

from cartppole.environment import Environment
from cartppole.policy import Policy


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
def evaluate(
    env_id: str = "CartPole-v1",
    checkpoint_path: Path = Path("checkpoints/policy.pt"),
    success_threshold: float = 475,
    n_episodes: int = 60,
    seed: int = 0,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    env = Environment(id=env_id, n_envs=1)
    policy = Policy(
        obs_dim=checkpoint["obs_dim"],
        act_dim=checkpoint["act_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    episode_returns = []

    try:
        for episode in range(n_episodes):
            reset = env.reset(seed=seed + episode)
            obs = tensor(reset.obs, dtype=torch.float32)
            total_reward = 0.0

            while True:
                with no_grad():
                    out = policy.get_action_and_value(obs)

                step = env.step(out.action.numpy())
                total_reward += float(step.reward[0])
                obs = tensor(step.obs, dtype=torch.float32)

                if bool(step.done[0]):
                    episode_returns.append(total_reward)
                    break
    finally:
        env.close()

    click.echo(f"{'checkpoint:':<24}{checkpoint_path}")
    click.echo(f"{'episodes:':<24}{n_episodes}")
    click.echo(
        f"{'return mean±std:':<24}{np.mean(episode_returns):.1f} ± {np.std(episode_returns):.1f}"
    )
    click.echo(f"{'return min:':<24}{np.min(episode_returns):.1f}")
    click.echo(f"{'return max:':<24}{np.max(episode_returns):.1f}")
    click.echo(
        f"{'success rate:':<24}{np.mean(np.array(episode_returns) >= success_threshold):.2f}  (return ≥ {success_threshold})"
    )


if __name__ == "__main__":
    evaluate()

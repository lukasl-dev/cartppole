from pathlib import Path

import click
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
@click.option("--seed", default=0, show_default=True, type=int)
def play(checkpoint_path: Path, env_id: str = "CartPole-v1", seed: int = 0) -> None:
    """Render one episode from a saved policy checkpoint.

    Args:
        checkpoint_path: Path to a checkpoint produced by ``train``.
        env_id: Gymnasium environment identifier to render.
        seed: Reset seed for the rendered episode.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    env = Environment(id=env_id, n_envs=1, render=True)
    policy = Policy(
        obs_dim=checkpoint["obs_dim"],
        act_dim=checkpoint["act_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    reset = env.reset(seed=seed)
    obs = tensor(reset.obs, dtype=torch.float32)
    total_reward = 0.0

    try:
        while True:
            with no_grad():
                out = policy.get_action_and_value(obs)

            step = env.step(out.action.numpy())
            total_reward += float(step.reward[0])
            obs = tensor(step.obs, dtype=torch.float32)

            if bool(step.done[0]):
                click.echo(f"episode return: {total_reward}")
                break
    finally:
        env.close()


if __name__ == "__main__":
    play()

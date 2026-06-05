from typing import NamedTuple

import click
from aim import Run
import torch
from torch.optim import Adam
import numpy as np

from cartppole.environment import Environment
from cartppole.policy import Policy


@click.command()
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--n-envs", default=8, show_default=True, type=int)
@click.option("--render", is_flag=True, help="Render the environment.")
@click.option("--hidden-dim", default=64, show_default=True, type=int)
@click.option("--learning-rate", default=2.5e-4, show_default=True, type=float)
def train(
    seed: int = 0,
    n_envs: int = 8,
    render: bool = False,
    hidden_dim: int = 64,
    learning_rate: float = 2.5e-4,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    run = Run()

    env = Environment(n_envs=n_envs, render=render)

    policy = Policy(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        hidden_dim=hidden_dim,
    )
    optim = Adam(policy.parameters(), lr=learning_rate)

    # TODO: implement
    ...

    env.close()


if __name__ == "__main__":
    train()

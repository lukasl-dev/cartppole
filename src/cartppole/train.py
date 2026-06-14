from pathlib import Path
import subprocess
from typing import Annotated, NamedTuple

import click
from aim import Run
from torch import Tensor, no_grad, tensor, zeros
import torch
from torch.optim import Adam
import numpy as np

from cartppole.environment import Environment
from cartppole.policy import Policy
from cartppole.advantages import (
    Advantage,
    generalised_advantage_estimation,
    monte_carlo,
)


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class Rollout(NamedTuple):
    obs: Annotated[Tensor, "n_steps n_envs obs_dim"]
    act: Annotated[Tensor, "n_steps n_envs"]
    log_prob: Annotated[Tensor, "n_steps n_envs"]
    rew: Annotated[Tensor, "n_steps n_envs"]
    done: Annotated[Tensor, "n_steps n_envs"]
    val: Annotated[Tensor, "n_steps n_envs"]
    next_val: Annotated[Tensor, "n_envs"]
    next_done: Annotated[Tensor, "n_envs"]


def rollout(
    seed: int,
    n_steps: int,
    env: Environment,
    policy: Policy,
) -> Rollout:
    reset = env.reset(seed=seed)
    next_obs: Annotated[Tensor, "n_envs obs_dim"] = tensor(
        reset.obs,
        dtype=torch.float32,
    )
    next_done: Annotated[Tensor, "n_envs"] = zeros(env.n_envs)

    obs: Annotated[Tensor, "n_steps n_envs obs_dim"] = zeros(
        (n_steps, env.n_envs, env.obs_dim)
    )
    act: Annotated[Tensor, "n_steps n_envs"] = zeros(
        (n_steps, env.n_envs),
        dtype=torch.long,
    )
    log_prob: Annotated[Tensor, "n_steps n_envs"] = zeros((n_steps, env.n_envs))
    rew: Annotated[Tensor, "n_steps n_envs"] = zeros((n_steps, env.n_envs))
    done: Annotated[Tensor, "n_steps n_envs"] = zeros((n_steps, env.n_envs))
    val: Annotated[Tensor, "n_steps n_envs"] = zeros((n_steps, env.n_envs))

    for step in range(n_steps):
        obs[step] = next_obs
        done[step] = next_done

        with no_grad():
            out = policy.get_action_and_value(next_obs)

        act[step] = out.action
        log_prob[step] = out.log_prob
        val[step] = out.val

        env_step = env.step(out.action.numpy())

        rew[step] = tensor(env_step.reward, dtype=torch.float32)
        next_done = tensor(env_step.done, dtype=torch.float32)

        if env.n_envs == 1 and bool(env_step.done[0]):
            reset = env.reset()
            next_obs = tensor(reset.obs, dtype=torch.float32)
        else:
            next_obs = tensor(env_step.obs, dtype=torch.float32)

    with no_grad():
        next_value: Annotated[Tensor, "n_envs"] = policy.get_value(next_obs)

    return Rollout(
        obs=obs,
        act=act,
        log_prob=log_prob,
        rew=rew,
        done=done,
        val=val,
        next_val=next_value,
        next_done=next_done,
    )


def collect_completed_episodes(
    roll: Rollout,
    current_ep_return: Annotated[Tensor, "n_envs"],
    current_ep_length: Annotated[Tensor, "n_envs"],
) -> tuple[list[float], list[int]]:
    n_steps = roll.rew.shape[0]
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    for step in range(n_steps):
        current_ep_return += roll.rew[step]
        current_ep_length += 1

        ended = roll.done[step + 1] if step < n_steps - 1 else roll.next_done

        for env_idx in range(len(current_ep_return)):
            if ended[env_idx]:
                episode_returns.append(float(current_ep_return[env_idx]))
                episode_lengths.append(int(current_ep_length[env_idx]))
                current_ep_return[env_idx] = 0
                current_ep_length[env_idx] = 0

    return episode_returns, episode_lengths


def normalise_advantages(
    adv: Annotated[Tensor, "batch_size"],
) -> Annotated[Tensor, "batch_size"]:
    r"""Normalise advantages.

    $$
    \hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}
    $$
    """
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def probability_ratio(
    log_prob: Annotated[Tensor, "batch_size"],
    old_log_prob: Annotated[Tensor, "batch_size"],
) -> Annotated[Tensor, "batch_size"]:
    r"""Compute the PPO probability ratio.

    $$
    r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\mathrm{old}}(a_t \mid s_t)}
    $$

    In log-space:

    $$
    r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_\mathrm{old}})
    $$
    """
    return (log_prob - old_log_prob).exp()


def clipped_policy_loss(
    ratio: Annotated[Tensor, "batch_size"],
    adv: Annotated[Tensor, "batch_size"],
    clip_coef: float,
) -> Annotated[Tensor, "scalar"]:
    r"""Compute the negative PPO clipped surrogate objective.

    PPO maximises:

    $$
    L^\mathrm{CLIP}(\theta) = \mathbb{E}_t\left[
        \min
        \left(
            r_t(\theta) A_t,
            \mathrm{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t
        \right)
    \right]
    $$

    Since PyTorch optimisers minimise losses, this function returns
    ``-L^CLIP``.
    """
    return torch.max(
        -adv * ratio,
        -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef),
    ).mean()


def value_function_loss(
    val: Annotated[Tensor, "batch_size"],
    ret: Annotated[Tensor, "batch_size"],
) -> Annotated[Tensor, "scalar"]:
    r"""Compute the critic value-function loss.

    $$
    L^\mathrm{VF}(\theta) = \frac{1}{2} \mathbb{E}_t \left[
        \left( V_\theta(s_t) - G_t \right)^2
    \right]
    $$
    """
    return 0.5 * ((val - ret) ** 2).mean()


def entropy_bonus(
    entropy: Annotated[Tensor, "batch_size"],
) -> Annotated[Tensor, "scalar"]:
    r"""Compute the mean policy entropy bonus.

    $$
    \mathcal{H}(\pi_\theta(\cdot \mid s_t)) =
        -\sum_a \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t)
    $$
    """
    return entropy.mean()


def ppo_loss(
    policy_loss: Annotated[Tensor, "scalar"],
    value_loss: Annotated[Tensor, "scalar"],
    entropy: Annotated[Tensor, "scalar"],
    value_coef: float,
    entropy_coef: float,
) -> Annotated[Tensor, "scalar"]:
    r"""Combine the PPO policy, value, and entropy losses.

    $$
    L = L^\mathrm{policy} + c_1 L^\mathrm{VF} - c_2 H[\pi_\theta]
    $$
    """
    return policy_loss + value_coef * value_loss - entropy_coef * entropy


@click.command()
@click.option(
    "--env",
    "env_id",
    default="CartPole-v1",
    show_default=True,
    type=str,
    help="Gymnasium environment ID.",
)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--n-envs", default=8, show_default=True, type=int)
@click.option("--render", is_flag=True, help="Render the environment.")
@click.option("--hidden-dim", default=64, show_default=True, type=int)
@click.option("--learning-rate", default=2.5e-4, show_default=True, type=float)
@click.option("--n-steps", default=128, show_default=True, type=int)
@click.option("--mini-batch-size", default=256, show_default=True, type=int)
@click.option("--clip-coef", default=0.2, show_default=True, type=float)
@click.option("--value-coef", default=0.5, show_default=True, type=float)
@click.option("--entropy-coef", default=0.01, show_default=True, type=float)
@click.option("--update-epochs", default=4, show_default=True, type=int)
@click.option("--total-timesteps", default=100_000, show_default=True, type=int)
@click.option(
    "--checkpoint-path",
    default="checkpoints/policy.pt",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--discount-factor",
    "--mc.discount_factor",
    "discount_factor",
    default=0.99,
    show_default=True,
    type=float,
    help="Reward discount factor gamma.",
)
@click.option(
    "--advantage-estimator",
    default="gae",
    show_default=True,
    type=click.Choice(["gae", "monte-carlo"]),
    help="Advantage estimator used for PPO targets.",
)
@click.option(
    "--gae-lambda",
    "--gae.lambda",
    "gae_lambda",
    default=0.95,
    show_default=True,
    type=float,
    help="Lambda parameter for Generalised Advantage Estimation.",
)
def train(
    env_id: str = "CartPole-v1",
    seed: int = 0,
    n_envs: int = 8,
    render: bool = False,
    hidden_dim: int = 64,
    learning_rate: float = 2.5e-4,
    n_steps: int = 128,
    mini_batch_size: int = 256,
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    update_epochs: int = 4,
    total_timesteps: int = 100_000,
    checkpoint_path: Path = Path("checkpoints/policy.pt"),
    advantage_estimator: str = "gae",
    discount_factor: float = 0.99,
    gae_lambda: float = 0.95,
) -> None:
    params = locals().copy()

    run = Run()
    run.add_tag(env_id)

    commit_hash = git_commit_hash()
    if commit_hash is not None:
        run.add_tag(commit_hash)
        run["commit"] = commit_hash

    for k, v in params.items():
        run[k] = str(v) if isinstance(v, Path) else v

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = Environment(id=env_id, n_envs=n_envs, render=render)
    policy = Policy(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        hidden_dim=hidden_dim,
    )
    optim = Adam(policy.parameters(), lr=learning_rate)

    n_updates = total_timesteps // (n_steps * n_envs)
    progress_interval = max(1, n_updates // 20)

    current_ep_return = zeros(n_envs)
    current_ep_length = zeros(n_envs, dtype=torch.long)

    for update in range(n_updates):
        if update % progress_interval == 0 or update == n_updates - 1:
            click.echo(
                f"\rRunning update {update + 1}/{n_updates}", nl=update == n_updates - 1
            )

        roll: Rollout = rollout(
            seed=seed + update,
            n_steps=n_steps,
            env=env,
            policy=policy,
        )

        adv: Advantage
        match advantage_estimator:
            case "gae":
                adv = generalised_advantage_estimation(
                    rew=roll.rew,
                    dones=roll.done,
                    val=roll.val,
                    next_val=roll.next_val,
                    next_done=roll.next_done,
                    discount_factor=discount_factor,
                    gae_lambda=gae_lambda,
                )
            case "monte-carlo":
                adv = monte_carlo(
                    rew=roll.rew,
                    dones=roll.done,
                    val=roll.val,
                    next_val=roll.next_val,
                    next_done=roll.next_done,
                    discount_factor=discount_factor,
                )
            case _:
                raise ValueError(f"unknown advantage estimator: {advantage_estimator}")

        batch_obs: Annotated[Tensor, "batch_size obs_dim"] = roll.obs.reshape(
            (-1, roll.obs.shape[-1])
        )
        batch_act: Annotated[Tensor, "batch_size"] = roll.act.reshape(-1)
        batch_log_prob: Annotated[Tensor, "batch_size"] = roll.log_prob.reshape(-1)
        batch_ret: Annotated[Tensor, "batch_size"] = adv.ret.reshape(-1)
        batch_adv: Annotated[Tensor, "batch_size"] = adv.adv.reshape(-1)

        batch_adv = normalise_advantages(batch_adv)

        batch_size = batch_act.shape[0]
        n_minibatches = batch_size // mini_batch_size

        episode_returns, episode_lengths = collect_completed_episodes(
            roll=roll,
            current_ep_return=current_ep_return,
            current_ep_length=current_ep_length,
        )

        if episode_returns:
            run.track(
                float(np.mean(episode_returns)), name="episode/return_mean", step=update
            )
            run.track(
                float(np.mean(episode_lengths)), name="episode/length_mean", step=update
            )
            success_rate = np.mean([r >= 475 for r in episode_returns])
            run.track(float(success_rate), name="episode/success_rate", step=update)

        run.track(float(roll.rew.mean()), name="rollout/reward_mean", step=update)
        run.track(float(roll.rew.sum()), name="rollout/reward_sum", step=update)
        run.track(float(adv.ret.mean()), name="rollout/return_mean", step=update)
        run.track(float(adv.adv.mean()), name="rollout/advantage_mean", step=update)
        run.track(float(roll.val.mean()), name="rollout/value_mean", step=update)

        for epoch in range(update_epochs):
            batch_perm = torch.randperm(batch_size)
            for mini_batch_start in range(0, batch_size, mini_batch_size):
                mini_batch = batch_perm[
                    mini_batch_start : mini_batch_start + mini_batch_size
                ]
                mini_batch_act = batch_act[mini_batch]
                mini_batch_obs = batch_obs[mini_batch]
                mini_batch_log_prob = batch_log_prob[mini_batch]
                mini_batch_adv = batch_adv[mini_batch]
                mini_batch_ret = batch_ret[mini_batch]

                out = policy.get_action_and_value(mini_batch_obs, mini_batch_act)
                ratio = probability_ratio(out.log_prob, mini_batch_log_prob)
                policy_loss = clipped_policy_loss(ratio, mini_batch_adv, clip_coef)
                value_loss = value_function_loss(out.val, mini_batch_ret)
                entropy = entropy_bonus(out.entropy)
                loss = ppo_loss(
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    entropy=entropy,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                )

                optim.zero_grad()
                loss.backward()
                optim.step()

                mini_batch_idx = mini_batch_start // mini_batch_size
                update_step = (
                    update * update_epochs + epoch
                ) * n_minibatches + mini_batch_idx

                run.track(float(loss.detach()), name="loss", step=update_step)
                run.track(
                    float(policy_loss.detach()),
                    name="loss/policy",
                    step=update_step,
                )
                run.track(
                    float(value_loss.detach()),
                    name="loss/value",
                    step=update_step,
                )
                run.track(
                    float(entropy.detach()),
                    name="policy/entropy",
                    step=update_step,
                )
                run.track(
                    float(ratio.mean().detach()),
                    name="policy/ratio_mean",
                    step=update_step,
                )
                run.track(
                    float(ratio.std().detach()),
                    name="policy/ratio_std",
                    step=update_step,
                )

    env.close()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "obs_dim": env.obs_dim,
            "act_dim": env.act_dim,
            "hidden_dim": hidden_dim,
        },
        checkpoint_path,
    )


if __name__ == "__main__":
    train()

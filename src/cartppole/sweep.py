from itertools import product
from pathlib import Path
from statistics import fmean, pstdev
from typing import NamedTuple

import click

from cartppole.evaluate import evaluate
from cartppole.train import Metric, PolicyLoss, train


class SweepResult(NamedTuple):
    """Configuration and evaluation statistics for one sweep run."""

    run_hash: str
    seed: int
    clip_coef: float
    learning_rate: float
    value_coef: float
    entropy_coef: float
    advantage_estimator: str
    normalise_advantages: bool
    policy_loss: str
    gae_lambda: float
    discount_factor: float
    n_steps: int
    update_epochs: int
    return_mean: float
    return_std: float
    success_rate: float
    checkpoint_path: Path


def parse_float_list(
    _: click.Context, param: click.Parameter, value: str
) -> list[float]:
    """Parse a comma-separated Click option as floats.

    Args:
        _: Click context supplied by the option callback.
        param: Click option being parsed, used for error messages.
        value: Comma-separated float values.

    Returns:
        Parsed floating-point values.

    Raises:
        click.BadParameter: If parsing fails or the list is empty.
    """
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise click.BadParameter("expected a comma-separated list", param=param)
    try:
        return [float(item) for item in values]
    except ValueError as err:
        raise click.BadParameter(
            "expected comma-separated floats", param=param
        ) from err


def parse_int_list(_: click.Context, param: click.Parameter, value: str) -> list[int]:
    """Parse a comma-separated Click option as integers.

    Args:
        _: Click context supplied by the option callback.
        param: Click option being parsed, used for error messages.
        value: Comma-separated integer values.

    Returns:
        Parsed integer values.

    Raises:
        click.BadParameter: If parsing fails or the list is empty.
    """
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise click.BadParameter("expected a comma-separated list", param=param)
    try:
        return [int(item) for item in values]
    except ValueError as err:
        raise click.BadParameter(
            "expected comma-separated integers", param=param
        ) from err


def parse_bool_list(_: click.Context, param: click.Parameter, value: str) -> list[bool]:
    """Parse a comma-separated Click option as booleans.

    Accepted truthy values are ``true``, ``t``, ``1``, ``yes``, and ``y``.
    Accepted falsey values are ``false``, ``f``, ``0``, ``no``, and ``n``.
    """
    values = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not values:
        raise click.BadParameter("expected a comma-separated list", param=param)

    parsed: list[bool] = []
    for item in values:
        match item:
            case "true" | "t" | "1" | "yes" | "y":
                parsed.append(True)
            case "false" | "f" | "0" | "no" | "n":
                parsed.append(False)
            case _:
                raise click.BadParameter(
                    "expected comma-separated booleans, e.g. true,false",
                    param=param,
                )

    return parsed


def parse_advantage_estimators(
    _: click.Context,
    param: click.Parameter,
    value: str,
) -> list[str]:
    """Parse advantage-estimator choices for a sweep.

    Args:
        _: Click context supplied by the option callback.
        param: Click option being parsed, used for error messages.
        value: Comma-separated estimator names.

    Returns:
        A validated list containing ``gae`` and/or ``mc``.
    """
    choices = {"gae", "mc"}
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise click.BadParameter("expected a comma-separated list", param=param)

    invalid = [item for item in values if item not in choices]
    if invalid:
        raise click.BadParameter(
            f"expected choices from {sorted(choices)}, got {invalid}",
            param=param,
        )

    return values


def parse_policy_losses(
    _: click.Context,
    param: click.Parameter,
    value: str,
) -> list[str]:
    """Parse PPO policy-loss choices for a sweep.

    Args:
        _: Click context supplied by the option callback.
        param: Click option being parsed, used for error messages.
        value: Comma-separated loss names.

    Returns:
        A validated list of ``PolicyLoss`` string values.
    """
    choices = {loss.value for loss in PolicyLoss}
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise click.BadParameter("expected a comma-separated list", param=param)

    invalid = [item for item in values if item not in choices]
    if invalid:
        raise click.BadParameter(
            f"expected choices from {sorted(choices)}, got {invalid}",
            param=param,
        )

    return values


def parse_rollout_update_ratios(
    _: click.Context,
    param: click.Parameter,
    value: str,
) -> list[tuple[int, int]]:
    """Parse ``N_STEPS:UPDATE_EPOCHS`` rollout/update pairs.

    Args:
        _: Click context supplied by the option callback.
        param: Click option being parsed, used for error messages.
        value: Comma-separated rollout/update pairs.

    Returns:
        Positive ``(n_steps, update_epochs)`` pairs.
    """
    ratios: list[tuple[int, int]] = []

    for item in [part.strip() for part in value.split(",") if part.strip()]:
        try:
            n_steps_raw, update_epochs_raw = item.split(":", maxsplit=1)
            n_steps, update_epochs = int(n_steps_raw), int(update_epochs_raw)
            if n_steps <= 0 or update_epochs <= 0:
                raise ValueError
            ratios.append((n_steps, update_epochs))
        except ValueError as err:
            raise click.BadParameter(
                "expected comma-separated N_STEPS:UPDATE_EPOCHS pairs, "
                "e.g. 128:4,256:4,128:8",
                param=param,
            ) from err

    if not ratios:
        raise click.BadParameter("expected at least one ratio", param=param)

    return ratios


def checkpoint_for_sweep(
    checkpoint_path: Path,
    seed: int,
    clip_coef: float,
    learning_rate: float,
    value_coef: float,
    entropy_coef: float,
    advantage_estimator: str,
    normalise_advantages: bool,
    policy_loss: str,
    gae_lambda: float,
    discount_factor: float,
    n_steps: int,
    update_epochs: int,
) -> Path:
    """Build a checkpoint path that encodes sweep hyperparameters.

    Args:
        checkpoint_path: Base checkpoint path supplied by the CLI.
        seed: Training seed.
        clip_coef: PPO clip coefficient.
        learning_rate: Adam learning rate.
        value_coef: Value-loss coefficient.
        entropy_coef: Entropy-bonus coefficient.
        advantage_estimator: Advantage estimator name.
        normalise_advantages: Whether advantages are normalised.
        policy_loss: Policy-loss variant.
        gae_lambda: GAE lambda value.
        discount_factor: Reward discount factor.
        n_steps: Rollout length per environment.
        update_epochs: PPO epochs per rollout.

    Returns:
        A checkpoint path with a deterministic hyperparameter suffix.
    """

    def format_value(value: float | int) -> str:
        """Make numeric values safe for checkpoint file names."""
        return str(value).replace(".", "p")

    suffix = (
        f"seed-{seed}_"
        f"clip-{format_value(clip_coef)}_"
        f"lr-{format_value(learning_rate)}_"
        f"vf-{format_value(value_coef)}_"
        f"ent-{format_value(entropy_coef)}_"
        f"adv-{advantage_estimator}_"
        f"norm-{normalise_advantages}_"
        f"policy-{policy_loss}_"
        f"gae-{format_value(gae_lambda)}_"
        f"gamma-{format_value(discount_factor)}_"
        f"steps-{n_steps}_"
        f"epochs-{update_epochs}"
    )
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}_{suffix}{checkpoint_path.suffix}"
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render rows as a simple left-aligned Markdown table.

    Args:
        headers: Column headers.
        rows: Table body rows, already converted to strings.

    Returns:
        A Markdown table suitable for terminal output or copying into notes.
    """
    widths = [
        max(len(row[column]) for row in [headers, *rows])
        for column in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        """Format one row using the precomputed column widths."""
        return (
            "| "
            + " | ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            )
            + " |"
        )

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(headers), separator, *map(format_row, rows)])


def runs_table(results: list[SweepResult]) -> str:
    """Render one table row per sweep run.

    Args:
        results: Per-run sweep results.

    Returns:
        A Markdown table containing configuration and evaluation statistics.
    """
    headers = [
        "run",
        "seed",
        "clip",
        "lr",
        "vf",
        "ent",
        "adv",
        "norm",
        "policy",
        "gae λ",
        "γ",
        "n_steps",
        "epochs",
        "return μ±σ",
        "success",
    ]
    rows = [
        [
            result.run_hash,
            str(result.seed),
            str(result.clip_coef),
            str(result.learning_rate),
            str(result.value_coef),
            str(result.entropy_coef),
            result.advantage_estimator,
            str(result.normalise_advantages),
            result.policy_loss,
            str(result.gae_lambda),
            str(result.discount_factor),
            str(result.n_steps),
            str(result.update_epochs),
            f"{result.return_mean:.1f} ± {result.return_std:.1f}",
            f"{result.success_rate:.2f}",
        ]
        for result in results
    ]
    return markdown_table(headers, rows)


def summary_table(results: list[SweepResult]) -> str:
    """Summarise sweep performance across seeds and hyperparameters.

    Args:
        results: Per-run sweep results.

    Returns:
        A Markdown table with grouped mean and standard deviation values.
    """
    grouped: dict[
        tuple[float, float, float, float, str, bool, str, float, float, int, int],
        list[SweepResult],
    ] = {}

    for result in results:
        key = (
            result.clip_coef,
            result.learning_rate,
            result.value_coef,
            result.entropy_coef,
            result.advantage_estimator,
            result.normalise_advantages,
            result.policy_loss,
            result.gae_lambda,
            result.discount_factor,
            result.n_steps,
            result.update_epochs,
        )
        grouped.setdefault(key, []).append(result)

    headers = [
        "runs",
        "seeds",
        "clip",
        "lr",
        "vf",
        "ent",
        "adv",
        "norm",
        "policy",
        "gae λ",
        "γ",
        "n_steps",
        "epochs",
        "return across seeds",
        "success across seeds",
    ]
    rows: list[list[str]] = []
    for (
        clip_coef,
        learning_rate,
        value_coef,
        entropy_coef,
        advantage_estimator,
        normalise_advantages,
        policy_loss,
        gae_lambda,
        discount_factor,
        n_steps,
        update_epochs,
    ), group in grouped.items():
        returns = [result.return_mean for result in group]
        success_rates = [result.success_rate for result in group]
        rows.append(
            [
                str(len(group)),
                ",".join(str(result.seed) for result in group),
                str(clip_coef),
                str(learning_rate),
                str(value_coef),
                str(entropy_coef),
                advantage_estimator,
                str(normalise_advantages),
                policy_loss,
                str(gae_lambda),
                str(discount_factor),
                str(n_steps),
                str(update_epochs),
                f"{fmean(returns):.1f} ± {pstdev(returns):.1f}",
                f"{fmean(success_rates):.2f} ± {pstdev(success_rates):.2f}",
            ]
        )

    return markdown_table(headers, rows)


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
    "--seeds",
    "--seed",
    default="0",
    show_default=True,
    callback=parse_int_list,
    help="Comma-separated training seeds to sweep.",
)
@click.option("--n-envs", default=8, show_default=True, type=int)
@click.option("--render", is_flag=True, help="Render the environment.")
@click.option("--hidden-dim", default=64, show_default=True, type=int)
@click.option(
    "--learning-rates",
    "--learning-rate",
    default="2.5e-4",
    show_default=True,
    callback=parse_float_list,
    help="Comma-separated learning rates to sweep.",
)
@click.option("--mini-batch-size", default=256, show_default=True, type=int)
@click.option(
    "--value-coefs",
    "--value-coef",
    default="0.5",
    show_default=True,
    callback=parse_float_list,
    help="Comma-separated value loss coefficients to sweep.",
)
@click.option(
    "--entropy-coefs",
    "--entropy-coef",
    default="0.01",
    show_default=True,
    callback=parse_float_list,
    help="Comma-separated entropy coefficients to sweep.",
)
@click.option("--total-timesteps", default=100_000, show_default=True, type=int)
@click.option("--n-eval-episodes", default=60, show_default=True, type=int)
@click.option("--success-threshold", default=475.0, show_default=True, type=float)
@click.option("--eval-seed", default=10_000, show_default=True, type=int)
@click.option(
    "--name",
    "sweep_name",
    default=None,
    type=str,
    help="Optional sweep name. Added as an Aim tag and parameter.",
)
@click.option(
    "--checkpoint-path",
    default="checkpoints/sweep/policy.pt",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Base checkpoint path. Each sweep run gets a suffixed file name.",
)
@click.option(
    "--discount-factors",
    "--discount-factor",
    "--mc.discount_factor",
    "discount_factors",
    default="0.99",
    show_default=True,
    callback=parse_float_list,
    help="Comma-separated reward discount factors gamma to sweep.",
)
@click.option(
    "--advantage-estimators",
    "--advantage-estimator",
    default="gae",
    show_default=True,
    callback=parse_advantage_estimators,
    help="Comma-separated advantage estimators to sweep: gae,mc.",
)
@click.option(
    "--normalise-advantages",
    default="true",
    show_default=True,
    callback=parse_bool_list,
    help="Comma-separated booleans for advantage normalisation, e.g. true,false.",
)
@click.option(
    "--policy-losses",
    "--policy-loss",
    default=PolicyLoss.clipped.value,
    show_default=True,
    callback=parse_policy_losses,
    help="Comma-separated policy losses to sweep: clipped,unclipped.",
)
@click.option(
    "--clip-coefs",
    "--clip-ranges",
    "clip_coefs",
    default="0.1,0.2,0.3",
    show_default=True,
    callback=parse_float_list,
    help="Comma-separated PPO clip coefficients to sweep.",
)
@click.option(
    "--gae-lambdas",
    default="0.9,0.95,0.97",
    show_default=True,
    callback=parse_float_list,
    help="Comma-separated GAE lambda values to sweep.",
)
@click.option(
    "--rollout-update-ratios",
    "--rollout-update-ratio",
    default="128:4",
    show_default=True,
    callback=parse_rollout_update_ratios,
    help="Comma-separated N_STEPS:UPDATE_EPOCHS pairs to sweep.",
)
def sweep(
    env_id: str = "CartPole-v1",
    seeds: list[int] | None = None,
    n_envs: int = 8,
    render: bool = False,
    hidden_dim: int = 64,
    learning_rates: list[float] | None = None,
    mini_batch_size: int = 256,
    value_coefs: list[float] | None = None,
    entropy_coefs: list[float] | None = None,
    total_timesteps: int = 100_000,
    n_eval_episodes: int = 60,
    success_threshold: float = 475,
    eval_seed: int = 10_000,
    sweep_name: str | None = None,
    checkpoint_path: Path = Path("checkpoints/sweep/policy.pt"),
    discount_factors: list[float] | None = None,
    advantage_estimators: list[str] | None = None,
    normalise_advantages: list[bool] | None = None,
    policy_losses: list[str] | None = None,
    clip_coefs: list[float] | None = None,
    gae_lambdas: list[float] | None = None,
    rollout_update_ratios: list[tuple[int, int]] | None = None,
) -> list[SweepResult]:
    """Train and evaluate a grid of PPO hyperparameter settings.

    The command forms the Cartesian product of requested seeds and
    hyperparameter lists, trains one checkpoint for each experiment,
    evaluates the checkpoint, logs summary metrics to Aim, and prints
    Markdown tables of the results.

    Returns:
        A list of per-run sweep results.
    """
    seeds = seeds or [0]
    clip_coefs = clip_coefs or [0.1, 0.2, 0.3]
    learning_rates = learning_rates or [2.5e-4]
    value_coefs = value_coefs or [0.5]
    entropy_coefs = entropy_coefs or [0.01]
    advantage_estimators = advantage_estimators or ["gae"]
    normalise_advantages = normalise_advantages or [True]
    policy_losses = policy_losses or [PolicyLoss.clipped.value]
    gae_lambdas = gae_lambdas or [0.9, 0.95, 0.97]
    discount_factors = discount_factors or [0.99]
    rollout_update_ratios = rollout_update_ratios or [(128, 4)]

    results: list[SweepResult] = []
    experiments = [
        (
            seed,
            clip_coef,
            learning_rate,
            value_coef,
            entropy_coef,
            advantage_estimator,
            normalise_advantage,
            policy_loss,
            gae_lambda,
            discount_factor,
            rollout_update_ratio,
        )
        for (
            seed,
            learning_rate,
            value_coef,
            entropy_coef,
            advantage_estimator,
            normalise_advantage,
            policy_loss,
            discount_factor,
            rollout_update_ratio,
        ) in product(
            seeds,
            learning_rates,
            value_coefs,
            entropy_coefs,
            advantage_estimators,
            normalise_advantages,
            policy_losses,
            discount_factors,
            rollout_update_ratios,
        )
        for clip_coef in (
            clip_coefs if policy_loss == PolicyLoss.clipped else [clip_coefs[0]]
        )
        for gae_lambda in (
            gae_lambdas if advantage_estimator == "gae" else [gae_lambdas[0]]
        )
    ]

    for index, (
        seed,
        clip_coef,
        learning_rate,
        value_coef,
        entropy_coef,
        advantage_estimator,
        normalise_advantage,
        policy_loss,
        gae_lambda,
        discount_factor,
        rollout_update_ratio,
    ) in enumerate(
        experiments,
        start=1,
    ):
        n_steps, update_epochs = rollout_update_ratio
        run_checkpoint_path = checkpoint_for_sweep(
            checkpoint_path=checkpoint_path,
            seed=seed,
            clip_coef=clip_coef,
            learning_rate=learning_rate,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            advantage_estimator=advantage_estimator,
            normalise_advantages=normalise_advantage,
            policy_loss=policy_loss,
            gae_lambda=gae_lambda,
            discount_factor=discount_factor,
            n_steps=n_steps,
            update_epochs=update_epochs,
        )

        click.echo(
            f"Sweep {index}/{len(experiments)}: "
            f"seed={seed}, "
            f"clip_coef={clip_coef}, "
            f"learning_rate={learning_rate}, "
            f"value_coef={value_coef}, "
            f"entropy_coef={entropy_coef}, "
            f"advantage_estimator={advantage_estimator}, "
            f"normalise_advantages={normalise_advantage}, "
            f"policy_loss={policy_loss}, "
            f"gae_lambda={gae_lambda}, "
            f"discount_factor={discount_factor}, "
            f"n_steps={n_steps}, "
            f"update_epochs={update_epochs}"
        )

        run = train(
            env_id=env_id,
            seed=seed,
            n_envs=n_envs,
            render=render,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            n_steps=n_steps,
            mini_batch_size=mini_batch_size,
            clip_coef=clip_coef,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            update_epochs=update_epochs,
            total_timesteps=total_timesteps,
            checkpoint_path=run_checkpoint_path,
            advantage_estimator=advantage_estimator,
            normalise_advantages=normalise_advantage,
            policy_loss=policy_loss,
            discount_factor=discount_factor,
            gae_lambda=gae_lambda,
        )
        try:
            run.add_tag("sweep")
            if sweep_name is not None:
                run.add_tag(sweep_name)
                run["sweep"] = sweep_name

            evaluation = evaluate(
                env_id=env_id,
                checkpoint_path=run_checkpoint_path,
                success_threshold=success_threshold,
                n_episodes=n_eval_episodes,
                seed=eval_seed,
            )
            run.track(evaluation.return_mean, name=Metric.eval_return_mean, step=0)
            run.track(evaluation.return_std, name=Metric.eval_return_std, step=0)
            run.track(evaluation.success_rate, name=Metric.eval_success_rate, step=0)
            results.append(
                SweepResult(
                    run_hash=run.hash,
                    seed=seed,
                    clip_coef=clip_coef,
                    learning_rate=learning_rate,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    advantage_estimator=advantage_estimator,
                    normalise_advantages=normalise_advantage,
                    policy_loss=policy_loss,
                    gae_lambda=gae_lambda,
                    discount_factor=discount_factor,
                    n_steps=n_steps,
                    update_epochs=update_epochs,
                    return_mean=evaluation.return_mean,
                    return_std=evaluation.return_std,
                    success_rate=evaluation.success_rate,
                    checkpoint_path=run_checkpoint_path,
                )
            )
        finally:
            run.close()

    title = "Sweep" if sweep_name is None else f"Sweep: {sweep_name}"
    click.echo(f"\n## {title} summary across seeds")
    click.echo(summary_table(results))
    click.echo(f"\n## {title} runs")
    click.echo(runs_table(results))
    return results


if __name__ == "__main__":
    sweep()

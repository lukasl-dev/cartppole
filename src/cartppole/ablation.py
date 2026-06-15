from itertools import product
from pathlib import Path
from typing import NamedTuple

import click

from cartppole.evaluate import evaluate
from cartppole.train import Metric, train


class AblationResult(NamedTuple):
    run_hash: str
    seed: int
    clip_coef: float
    learning_rate: float
    value_coef: float
    entropy_coef: float
    advantage_estimator: str
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
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise click.BadParameter("expected a comma-separated list", param=param)
    try:
        return [int(item) for item in values]
    except ValueError as err:
        raise click.BadParameter(
            "expected comma-separated integers", param=param
        ) from err


def parse_advantage_estimators(
    _: click.Context,
    param: click.Parameter,
    value: str,
) -> list[str]:
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


def parse_rollout_update_ratios(
    _: click.Context,
    param: click.Parameter,
    value: str,
) -> list[tuple[int, int]]:
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


def checkpoint_for_ablation(
    checkpoint_path: Path,
    seed: int,
    clip_coef: float,
    learning_rate: float,
    value_coef: float,
    entropy_coef: float,
    advantage_estimator: str,
    gae_lambda: float,
    discount_factor: float,
    n_steps: int,
    update_epochs: int,
) -> Path:
    def format_value(value: float | int) -> str:
        return str(value).replace(".", "p")

    suffix = (
        f"seed-{seed}_"
        f"clip-{format_value(clip_coef)}_"
        f"lr-{format_value(learning_rate)}_"
        f"vf-{format_value(value_coef)}_"
        f"ent-{format_value(entropy_coef)}_"
        f"adv-{advantage_estimator}_"
        f"gae-{format_value(gae_lambda)}_"
        f"gamma-{format_value(discount_factor)}_"
        f"steps-{n_steps}_"
        f"epochs-{update_epochs}"
    )
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}_{suffix}{checkpoint_path.suffix}"
    )


def markdown_table(results: list[AblationResult]) -> str:
    headers = [
        "run",
        "seed",
        "clip",
        "lr",
        "vf",
        "ent",
        "adv",
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
            str(result.gae_lambda),
            str(result.discount_factor),
            str(result.n_steps),
            str(result.update_epochs),
            f"{result.return_mean:.1f} ± {result.return_std:.1f}",
            f"{result.success_rate:.2f}",
        ]
        for result in results
    ]
    widths = [
        max(len(row[column]) for row in [headers, *rows])
        for column in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return (
            "| "
            + " | ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            )
            + " |"
        )

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(headers), separator, *map(format_row, rows)])


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
    help="Comma-separated training seeds to ablate.",
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
    help="Comma-separated learning rates to ablate.",
)
@click.option("--mini-batch-size", default=256, show_default=True, type=int)
@click.option(
    "--value-coefs",
    "--value-coef",
    default="0.5",
    show_default=True,
    callback=parse_float_list,
    help="Comma-separated value loss coefficients to ablate.",
)
@click.option(
    "--entropy-coefs",
    "--entropy-coef",
    default="0.01",
    show_default=True,
    callback=parse_float_list,
    help="Comma-separated entropy coefficients to ablate.",
)
@click.option("--total-timesteps", default=100_000, show_default=True, type=int)
@click.option("--n-eval-episodes", default=60, show_default=True, type=int)
@click.option("--success-threshold", default=475.0, show_default=True, type=float)
@click.option("--eval-seed", default=10_000, show_default=True, type=int)
@click.option(
    "--name",
    "ablation_name",
    default=None,
    type=str,
    help="Optional ablation name. Added as an Aim tag and parameter.",
)
@click.option(
    "--checkpoint-path",
    default="checkpoints/ablation/policy.pt",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Base checkpoint path. Each ablation run gets a suffixed file name.",
)
@click.option(
    "--discount-factors",
    "--discount-factor",
    "--mc.discount_factor",
    "discount_factors",
    default="0.99",
    show_default=True,
    callback=parse_float_list,
    help="Comma-separated reward discount factors gamma to ablate.",
)
@click.option(
    "--advantage-estimators",
    "--advantage-estimator",
    default="gae",
    show_default=True,
    callback=parse_advantage_estimators,
    help="Comma-separated advantage estimators to ablate: gae,mc.",
)
@click.option(
    "--clip-coefs",
    "--clip-ranges",
    "clip_coefs",
    default="0.1,0.2,0.3",
    show_default=True,
    callback=parse_float_list,
    help="Comma-separated PPO clip coefficients to ablate.",
)
@click.option(
    "--gae-lambdas",
    default="0.9,0.95,0.97",
    show_default=True,
    callback=parse_float_list,
    help="Comma-separated GAE lambda values to ablate.",
)
@click.option(
    "--rollout-update-ratios",
    "--rollout-update-ratio",
    default="128:4",
    show_default=True,
    callback=parse_rollout_update_ratios,
    help="Comma-separated N_STEPS:UPDATE_EPOCHS pairs to ablate.",
)
def ablation(
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
    ablation_name: str | None = None,
    checkpoint_path: Path = Path("checkpoints/ablation/policy.pt"),
    discount_factors: list[float] | None = None,
    advantage_estimators: list[str] | None = None,
    clip_coefs: list[float] | None = None,
    gae_lambdas: list[float] | None = None,
    rollout_update_ratios: list[tuple[int, int]] | None = None,
) -> list[AblationResult]:
    seeds = seeds or [0]
    clip_coefs = clip_coefs or [0.1, 0.2, 0.3]
    learning_rates = learning_rates or [2.5e-4]
    value_coefs = value_coefs or [0.5]
    entropy_coefs = entropy_coefs or [0.01]
    advantage_estimators = advantage_estimators or ["gae"]
    gae_lambdas = gae_lambdas or [0.9, 0.95, 0.97]
    discount_factors = discount_factors or [0.99]
    rollout_update_ratios = rollout_update_ratios or [(128, 4)]

    results: list[AblationResult] = []
    experiments = [
        (
            seed,
            clip_coef,
            learning_rate,
            value_coef,
            entropy_coef,
            advantage_estimator,
            gae_lambda,
            discount_factor,
            rollout_update_ratio,
        )
        for (
            seed,
            clip_coef,
            learning_rate,
            value_coef,
            entropy_coef,
            advantage_estimator,
            discount_factor,
            rollout_update_ratio,
        ) in product(
            seeds,
            clip_coefs,
            learning_rates,
            value_coefs,
            entropy_coefs,
            advantage_estimators,
            discount_factors,
            rollout_update_ratios,
        )
        for gae_lambda in (gae_lambdas if advantage_estimator == "gae" else [gae_lambdas[0]])
    ]

    for index, (
        seed,
        clip_coef,
        learning_rate,
        value_coef,
        entropy_coef,
        advantage_estimator,
        gae_lambda,
        discount_factor,
        rollout_update_ratio,
    ) in enumerate(
        experiments,
        start=1,
    ):
        n_steps, update_epochs = rollout_update_ratio
        run_checkpoint_path = checkpoint_for_ablation(
            checkpoint_path=checkpoint_path,
            seed=seed,
            clip_coef=clip_coef,
            learning_rate=learning_rate,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            advantage_estimator=advantage_estimator,
            gae_lambda=gae_lambda,
            discount_factor=discount_factor,
            n_steps=n_steps,
            update_epochs=update_epochs,
        )

        click.echo(
            f"Ablation {index}/{len(experiments)}: "
            f"seed={seed}, "
            f"clip_coef={clip_coef}, "
            f"learning_rate={learning_rate}, "
            f"value_coef={value_coef}, "
            f"entropy_coef={entropy_coef}, "
            f"advantage_estimator={advantage_estimator}, "
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
            discount_factor=discount_factor,
            gae_lambda=gae_lambda,
        )
        try:
            run.add_tag("ablation")
            if ablation_name is not None:
                run.add_tag(ablation_name)
                run["ablation/name"] = ablation_name

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
                AblationResult(
                    run_hash=run.hash,
                    seed=seed,
                    clip_coef=clip_coef,
                    learning_rate=learning_rate,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    advantage_estimator=advantage_estimator,
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

    title = "Ablation runs" if ablation_name is None else f"Ablation runs: {ablation_name}"
    click.echo(f"\n## {title}")
    click.echo(markdown_table(results))
    return results


if __name__ == "__main__":
    ablation()

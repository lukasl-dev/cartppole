from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np

from cartppole.evaluate import evaluate
from cartppole.sweep import parse_int_list
from cartppole.train import Metric, PolicyLoss, train


@dataclass(frozen=True)
class AblationConfig:
    """Configuration switches that define one PPO ablation variant.

    The baseline configuration is copied and selectively modified to isolate
    the effect of one PPO component, such as GAE, entropy regularisation,
    value learning, clipping, or advantage normalisation.
    """

    advantage_estimator: str
    normalise_advantages: bool
    policy_loss: str
    value_coef: float
    entropy_coef: float


class AblationResult(NamedTuple):
    """Evaluation summary for one trained ablation run.

    Each result records the ablated configuration, the training seed, and
    the post-training evaluation statistics needed for per-run and
    across-seed reporting.
    """

    run_hash: str
    variant: str
    seed: int
    advantage_estimator: str
    normalise_advantages: bool
    policy_loss: str
    value_coef: float
    entropy_coef: float
    return_mean: float
    return_std: float
    success_rate: float


VARIANTS = {
    "baseline",
    "no_gae",
    "no_entropy_bonus",
    "no_value_loss",
    "no_clipping",
    "no_advantage_normalisation",
}


def parse_variants(
    _: click.Context,
    param: click.Parameter,
    value: str,
) -> list[str]:
    """Parse and validate a comma-separated list of ablation variants.

    Args:
        _: Click context supplied by the option callback.
        param: Click option being parsed, used for error messages.
        value: Comma-separated variant names.

    Returns:
        The validated variant names in the order requested by the user.

    Raises:
        click.BadParameter: If the list is empty or contains unknown names.
    """
    variants = [part.strip() for part in value.split(",") if part.strip()]
    if not variants:
        raise click.BadParameter("expected a comma-separated list", param=param)

    invalid = [variant for variant in variants if variant not in VARIANTS]
    if invalid:
        raise click.BadParameter(
            f"expected choices from {sorted(VARIANTS)}, got {invalid}",
            param=param,
        )

    return variants


def config_for_variant(variant: str, baseline: AblationConfig) -> AblationConfig:
    """Return the PPO configuration for an ablation variant.

    Args:
        variant: Name of the ablation variant to apply.
        baseline: Reference configuration from which variants are derived.

    Returns:
        The baseline configuration or a modified copy for the requested
        ablation.

    Raises:
        ValueError: If ``variant`` is not recognised.
    """
    match variant:
        case "baseline":
            return baseline
        case "no_gae":
            return replace(baseline, advantage_estimator="mc")
        case "no_entropy_bonus":
            return replace(baseline, entropy_coef=0.0)
        case "no_value_loss":
            return replace(baseline, value_coef=0.0)
        case "no_clipping":
            return replace(baseline, policy_loss=PolicyLoss.unclipped.value)
        case "no_advantage_normalisation":
            return replace(baseline, normalise_advantages=False)
        case _:
            raise ValueError(f"unknown ablation variant: {variant}")


def checkpoint_for_ablation(
    checkpoint_path: Path,
    variant: str,
    seed: int,
) -> Path:
    """Build the checkpoint path for one ablation run.

    Args:
        checkpoint_path: Base checkpoint path supplied by the CLI.
        variant: Ablation variant name.
        seed: Training seed.

    Returns:
        A path with the variant and seed encoded in the file name.
    """
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}_{variant}_seed-{seed}{checkpoint_path.suffix}"
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


def runs_table(results: list[AblationResult]) -> str:
    """Render one table row per ablation run.

    Args:
        results: Per-run ablation results.

    Returns:
        A Markdown table containing configuration and evaluation statistics.
    """
    return markdown_table(
        headers=[
            "run",
            "variant",
            "seed",
            "adv",
            "norm",
            "policy",
            "vf",
            "ent",
            "return μ±σ",
            "success",
        ],
        rows=[
            [
                result.run_hash,
                result.variant,
                str(result.seed),
                result.advantage_estimator,
                str(result.normalise_advantages),
                result.policy_loss,
                str(result.value_coef),
                str(result.entropy_coef),
                f"{result.return_mean:.1f} ± {result.return_std:.1f}",
                f"{result.success_rate:.2f}",
            ]
            for result in results
        ],
    )


def summary_table(results: list[AblationResult]) -> str:
    """Summarise ablation performance across seeds.

    Args:
        results: Per-run ablation results.

    Returns:
        A Markdown table with mean and standard deviation by variant.
    """
    rows: list[list[str]] = []
    variants = list(dict.fromkeys(result.variant for result in results))

    for variant in variants:
        variant_results = [result for result in results if result.variant == variant]
        returns = np.array([result.return_mean for result in variant_results])
        success_rates = np.array([result.success_rate for result in variant_results])
        rows.append(
            [
                variant,
                str(len(variant_results)),
                f"{np.mean(returns):.1f} ± {np.std(returns):.1f}",
                f"{np.mean(success_rates):.2f} ± {np.std(success_rates):.2f}",
            ]
        )

    return markdown_table(
        headers=["variant", "runs", "return across seeds", "success across seeds"],
        rows=rows,
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
    "--seeds",
    "--seed",
    default="0,1,2,3407",
    show_default=True,
    callback=parse_int_list,
    help="Comma-separated training seeds.",
)
@click.option(
    "--variants",
    default="baseline,no_gae,no_entropy_bonus,no_value_loss,no_clipping,no_advantage_normalisation",
    show_default=True,
    callback=parse_variants,
    help="Comma-separated ablation variants.",
)
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
    type=click.Choice(["gae", "mc"]),
    help="Baseline advantage estimator.",
)
@click.option(
    "--normalise-advantages/--no-normalise-advantages",
    default=True,
    show_default=True,
    help="Baseline advantage normalisation setting.",
)
@click.option(
    "--policy-loss",
    default=PolicyLoss.clipped.value,
    show_default=True,
    type=click.Choice([loss.value for loss in PolicyLoss]),
    help="Baseline policy loss.",
)
@click.option(
    "--gae-lambda",
    "--gae.lambda",
    "gae_lambda",
    default=0.95,
    show_default=True,
    type=float,
    help="Baseline GAE lambda.",
)
def ablation(
    env_id: str = "CartPole-v1",
    seeds: list[int] | None = None,
    variants: list[str] | None = None,
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
    n_eval_episodes: int = 60,
    success_threshold: float = 475,
    eval_seed: int = 10_000,
    ablation_name: str | None = None,
    checkpoint_path: Path = Path("checkpoints/ablation/policy.pt"),
    discount_factor: float = 0.99,
    advantage_estimator: str = "gae",
    normalise_advantages: bool = True,
    policy_loss: str = PolicyLoss.clipped,
    gae_lambda: float = 0.95,
) -> list[AblationResult]:
    """Train and evaluate PPO ablation variants.

    The command starts from a baseline PPO configuration, applies each
    requested variant, trains one policy per variant and seed, evaluates the
    saved checkpoint, logs metrics to Aim, and prints Markdown summaries.

    Returns:
        A list of per-run ablation results.
    """
    seeds = seeds or [0, 1, 2, 3407]
    variants = variants or sorted(VARIANTS)
    baseline = AblationConfig(
        advantage_estimator=advantage_estimator,
        normalise_advantages=normalise_advantages,
        policy_loss=policy_loss,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
    )

    results: list[AblationResult] = []
    total_runs = len(seeds) * len(variants)

    for index, (variant, seed) in enumerate(
        ((variant, seed) for variant in variants for seed in seeds),
        start=1,
    ):
        config = config_for_variant(variant, baseline)
        run_checkpoint_path = checkpoint_for_ablation(
            checkpoint_path=checkpoint_path,
            variant=variant,
            seed=seed,
        )

        click.echo(
            f"Ablation {index}/{total_runs}: "
            f"variant={variant}, "
            f"seed={seed}, "
            f"advantage_estimator={config.advantage_estimator}, "
            f"normalise_advantages={config.normalise_advantages}, "
            f"policy_loss={config.policy_loss}, "
            f"value_coef={config.value_coef}, "
            f"entropy_coef={config.entropy_coef}"
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
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            update_epochs=update_epochs,
            total_timesteps=total_timesteps,
            checkpoint_path=run_checkpoint_path,
            advantage_estimator=config.advantage_estimator,
            normalise_advantages=config.normalise_advantages,
            policy_loss=config.policy_loss,
            discount_factor=discount_factor,
            gae_lambda=gae_lambda,
        )
        try:
            run.add_tag("ablation")
            run.add_tag(variant)
            run["ablation/variant"] = variant
            if ablation_name is not None:
                run.add_tag(ablation_name)
                run["ablation"] = ablation_name

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
                    variant=variant,
                    seed=seed,
                    advantage_estimator=config.advantage_estimator,
                    normalise_advantages=config.normalise_advantages,
                    policy_loss=config.policy_loss,
                    value_coef=config.value_coef,
                    entropy_coef=config.entropy_coef,
                    return_mean=evaluation.return_mean,
                    return_std=evaluation.return_std,
                    success_rate=evaluation.success_rate,
                )
            )
        finally:
            run.close()

    title = "Ablation" if ablation_name is None else f"Ablation: {ablation_name}"
    click.echo(f"\n## {title} summary")
    click.echo(summary_table(results))
    click.echo(f"\n## {title} runs")
    click.echo(runs_table(results))
    return results


if __name__ == "__main__":
    ablation()

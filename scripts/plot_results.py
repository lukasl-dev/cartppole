#!/usr/bin/env python3
"""Plot CartPPOle metrics from an Aim repository.

The training and sweep scripts log metrics and hyperparameters to Aim. This
script turns those logs into reproducible static figures, so the report does not
only depend on screenshots from the Aim UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import csv

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from aim import Repo  # noqa: E402


@dataclass(frozen=True)
class RunSeries:
    """One Aim metric time series for a single run."""
    run_hash: str
    label: str
    steps: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class FinalValue:
    """Final metric value assigned to a plotting group."""
    run_hash: str
    group: str
    value: float


def aim_root(path: Path) -> Path:
    """Return the project root expected by ``aim.Repo``.

    Aim's Python API expects the directory containing ``.aim``. Passing the
    ``.aim`` directory itself would make Aim look for ``.aim/.aim``.
    """
    if path.name == ".aim":
        return path.parent
    return path


def parse_param_filters(filters: Iterable[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` filters supplied on the command line.

    Args:
        filters: Raw parameter filter strings.

    Returns:
        Mapping from run parameter names to expected string values.

    Raises:
        click.BadParameter: If any filter is not a ``KEY=VALUE`` pair.
    """
    parsed: dict[str, str] = {}
    for item in filters:
        if "=" not in item:
            raise click.BadParameter(
                f"parameter filters must be KEY=VALUE pairs, got {item!r}"
            )
        key, value = item.split("=", maxsplit=1)
        parsed[key.strip()] = value.strip()
    return parsed


def run_param(run: Any, key: str) -> Any:
    """Read one Aim run parameter defensively.

    Args:
        run: Aim run object.
        key: Parameter key.

    Returns:
        The parameter value, or ``None`` if Aim cannot read it.
    """
    try:
        return run.get(key)
    except Exception:
        return None


def run_matches(run: Any, tags: tuple[str, ...], params: dict[str, str]) -> bool:
    """Return whether an Aim run satisfies tag and parameter filters.

    Args:
        run: Aim run object.
        tags: Required Aim tags.
        params: Required parameter values as strings.

    Returns:
        ``True`` when all tags and parameter filters match.
    """
    run_tags = set(run.tags or [])
    if not set(tags).issubset(run_tags):
        return False

    for key, expected in params.items():
        actual = run_param(run, key)
        if str(actual) != expected:
            return False

    return True


def metric_data(run: Any, metric_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract one scalar Aim metric as ``(steps, values)`` arrays."""
    for name, context, _ in run.iter_metrics_info():
        if name != metric_name:
            continue

        metric = run.get_metric(name, context)
        if metric is None:
            return None

        steps, columns = metric.data.numpy()
        values = columns[0]
        return np.asarray(steps, dtype=float), np.asarray(values, dtype=float)

    return None


def short_hash(run_hash: str) -> str:
    """Return the short display form of an Aim run hash."""
    return run_hash[:8]


def value_label(value: Any) -> str:
    """Format a parameter value for a plot label."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def group_label(run: Any, keys: tuple[str, ...]) -> str:
    """Build a group label from selected Aim run parameters.

    Args:
        run: Aim run object.
        keys: Parameter names to include.

    Returns:
        A comma-separated label such as ``clip_coef=0.2, seed=0``.
    """
    parts = []
    for key in keys:
        value = run_param(run, key)
        parts.append(f"{key}={value_label(value)}")
    return ", ".join(parts)


def run_label(run: Any, label_params: tuple[str, ...]) -> str:
    """Build a legend label for an Aim run.

    Args:
        run: Aim run object.
        label_params: Parameter names to include before the seed and hash.

    Returns:
        A concise label identifying the run in line plots.
    """
    params = [group_label(run, label_params)] if label_params else []
    seed = run_param(run, "seed")
    if seed is not None:
        params.append(f"seed={seed}")
    params.append(short_hash(run.hash))
    return " | ".join(part for part in params if part)


def load_series(
    repo_path: Path,
    metric_name: str,
    tags: tuple[str, ...],
    params: dict[str, str],
    label_params: tuple[str, ...],
) -> list[RunSeries]:
    """Load matching metric time series from an Aim repository.

    Args:
        repo_path: Project root or ``.aim`` directory.
        metric_name: Aim metric to load.
        tags: Required Aim tags.
        params: Required parameter filters.
        label_params: Parameters included in run labels.

    Returns:
        Sorted metric series for matching runs.
    """
    repo = Repo(str(aim_root(repo_path)))
    series: list[RunSeries] = []
    try:
        for run in repo.iter_runs():
            if not run_matches(run, tags=tags, params=params):
                continue

            data = metric_data(run, metric_name)
            if data is None:
                continue

            steps, values = data
            if len(steps) == 0:
                continue

            order = np.argsort(steps)
            series.append(
                RunSeries(
                    run_hash=run.hash,
                    label=run_label(run, label_params),
                    steps=steps[order],
                    values=values[order],
                )
            )
    finally:
        repo.close()

    return sorted(series, key=lambda item: item.label)


def common_grid(series: list[RunSeries]) -> np.ndarray:
    """Choose a shared x-axis grid for aggregating metric series.

    Args:
        series: Metric series to aggregate.

    Returns:
        Shared step values when available, otherwise the first run's step
        grid for interpolation.
    """
    if not series:
        return np.array([])

    common_steps = set(series[0].steps.astype(int).tolist())
    for item in series[1:]:
        common_steps &= set(item.steps.astype(int).tolist())

    if common_steps:
        return np.asarray(sorted(common_steps), dtype=float)

    # Fall back to interpolating on the first run's grid when logs do not share
    # exact step indices.
    return series[0].steps


def smooth_values(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a centred moving average to a one-dimensional series.

    Args:
        values: Values to smooth.
        window: Moving-average window length.

    Returns:
        Smoothed values, or the original values when the window is inactive.
    """
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def write_metric_csv(path: Path, series: list[RunSeries]) -> None:
    """Write line-plot source data to CSV.

    Args:
        path: Output CSV path.
        series: Metric series to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "label", "step", "value"])
        for item in series:
            for step, value in zip(item.steps, item.values, strict=True):
                writer.writerow([item.run_hash, item.label, int(step), float(value)])


def write_ablation_csv(path: Path, values: list[FinalValue]) -> None:
    """Write grouped final-value statistics to CSV.

    Args:
        path: Output CSV path.
        values: Final values grouped by plot label.
    """
    grouped: dict[str, list[float]] = {}
    for item in values:
        grouped.setdefault(item.group, []).append(item.value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "runs", "mean", "std", "values"])
        for group, group_values in grouped.items():
            arr = np.asarray(group_values, dtype=float)
            writer.writerow(
                [
                    group,
                    len(group_values),
                    float(arr.mean()),
                    float(arr.std()),
                    ";".join(f"{value:g}" for value in group_values),
                ]
            )


def default_csv_path(output: Path) -> Path:
    """Return the default CSV path corresponding to a plot path."""
    return output.with_suffix(".csv")


@click.group()
def cli() -> None:
    """Create static plots from Aim logs."""


@cli.command()
@click.option(
    "--repo",
    "repo_path",
    default=".",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Project root containing .aim, or the .aim directory itself.",
)
@click.option("--metric", "metric_name", default="loss", show_default=True)
@click.option("--tag", "tags", multiple=True, help="Require an Aim tag.")
@click.option(
    "--param",
    "param_filters",
    multiple=True,
    help="Require an Aim run parameter, e.g. --param n_steps=256.",
)
@click.option(
    "--label-param",
    "label_params",
    multiple=True,
    help="Include a run parameter in legend labels.",
)
@click.option("--output", default="plots/metric.png", show_default=True, type=Path)
@click.option("--csv-output", default=None, type=Path)
@click.option("--aggregate/--no-aggregate", default=False, show_default=True)
@click.option("--moving-average", default=1, show_default=True, type=int)
@click.option("--max-runs", default=0, show_default=True, type=int)
@click.option("--title", default=None)
def metric(
    repo_path: Path,
    metric_name: str,
    tags: tuple[str, ...],
    param_filters: tuple[str, ...],
    label_params: tuple[str, ...],
    output: Path,
    csv_output: Path | None,
    aggregate: bool,
    moving_average: int,
    max_runs: int,
    title: str | None,
) -> None:
    """Plot a metric trace, optionally mean ± std across matched runs."""
    params = parse_param_filters(param_filters)
    series = load_series(
        repo_path=repo_path,
        metric_name=metric_name,
        tags=tags,
        params=params,
        label_params=label_params,
    )
    if max_runs > 0:
        series = series[:max_runs]
    if not series:
        raise click.ClickException("no matching Aim runs with the requested metric")

    output.parent.mkdir(parents=True, exist_ok=True)
    write_metric_csv(csv_output or default_csv_path(output), series)

    plt.figure(figsize=(8, 4.5))
    if aggregate and len(series) > 1:
        x_grid = common_grid(series)
        ys = []
        for item in series:
            y = np.interp(x_grid, item.steps, item.values)
            ys.append(smooth_values(y, moving_average))
        stack = np.vstack(ys)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        plt.plot(x_grid, mean, label="mean", linewidth=2)
        plt.fill_between(x_grid, mean - std, mean + std, alpha=0.2, label="± std")
    else:
        for item in series:
            plt.plot(
                item.steps,
                smooth_values(item.values, moving_average),
                label=item.label,
                alpha=0.85,
            )

    plt.xlabel("step")
    plt.ylabel(metric_name)
    plt.title(title or metric_name)
    plt.grid(alpha=0.25)
    if (aggregate and len(series) > 1) or len(series) <= 12:
        plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    click.echo(f"wrote {output}")


@cli.command()
@click.option(
    "--repo",
    "repo_path",
    default=".",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option("--metric", "metric_name", default="eval/return_mean", show_default=True)
@click.option("--tag", "tags", multiple=True, help="Require an Aim tag.")
@click.option(
    "--param",
    "param_filters",
    multiple=True,
    help="Require an Aim run parameter, e.g. --param total_timesteps=1000000.",
)
@click.option(
    "--group-by",
    "group_keys",
    multiple=True,
    required=True,
    help="Run parameter used as a bar-group label. Can be repeated.",
)
@click.option("--output", default="plots/ablation.png", show_default=True, type=Path)
@click.option("--csv-output", default=None, type=Path)
@click.option("--title", default=None)
def ablation(
    repo_path: Path,
    metric_name: str,
    tags: tuple[str, ...],
    param_filters: tuple[str, ...],
    group_keys: tuple[str, ...],
    output: Path,
    csv_output: Path | None,
    title: str | None,
) -> None:
    """Plot final metric mean ± std grouped by hyperparameter values."""
    params = parse_param_filters(param_filters)
    repo = Repo(str(aim_root(repo_path)))
    values: list[FinalValue] = []
    try:
        for run in repo.iter_runs():
            if not run_matches(run, tags=tags, params=params):
                continue
            data = metric_data(run, metric_name)
            if data is None:
                continue
            _, metric_values = data
            if len(metric_values) == 0:
                continue
            values.append(
                FinalValue(
                    run_hash=run.hash,
                    group=group_label(run, group_keys),
                    value=float(metric_values[-1]),
                )
            )
    finally:
        repo.close()

    if not values:
        raise click.ClickException("no matching Aim runs with the requested metric")

    write_ablation_csv(csv_output or default_csv_path(output), values)

    grouped: dict[str, list[float]] = {}
    for item in values:
        grouped.setdefault(item.group, []).append(item.value)

    labels = list(grouped)
    means = np.asarray([np.mean(grouped[label]) for label in labels], dtype=float)
    stds = np.asarray([np.std(grouped[label]) for label in labels], dtype=float)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6.0, 1.15 * len(labels)), 4.5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=5, color="#4C78A8", alpha=0.9)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(metric_name)
    plt.title(title or metric_name)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    click.echo(f"wrote {output}")


@cli.command(name="all")
@click.option(
    "--repo",
    "repo_path",
    default=".",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option("--output-dir", default="plots", show_default=True, type=Path)
@click.option(
    "--sweep-suffix",
    default="1000000",
    show_default=True,
    help="Suffix used by task_ablation tags, e.g. 1000000.",
)
def plot_all(repo_path: Path, output_dir: Path, sweep_suffix: str) -> None:
    """Generate the standard submission plots from the task ablation Aim runs."""
    ctx = click.get_current_context()
    ctx.invoke(
        ablation,
        repo_path=repo_path,
        metric_name="eval/return_mean",
        tags=(f"task-clip-range-{sweep_suffix}",),
        param_filters=(),
        group_keys=("clip_coef",),
        output=output_dir / "clip_range_return.png",
        csv_output=None,
        title="Clip range ablation",
    )
    ctx.invoke(
        ablation,
        repo_path=repo_path,
        metric_name="eval/return_mean",
        tags=(f"task-gae-lambda-{sweep_suffix}",),
        param_filters=(),
        group_keys=("advantage_estimator", "gae_lambda"),
        output=output_dir / "advantage_return.png",
        csv_output=None,
        title="Advantage estimator / GAE lambda ablation",
    )
    ctx.invoke(
        ablation,
        repo_path=repo_path,
        metric_name="eval/return_mean",
        tags=(f"task-rollout-update-{sweep_suffix}",),
        param_filters=(),
        group_keys=("n_steps", "update_epochs"),
        output=output_dir / "rollout_update_return.png",
        csv_output=None,
        title="Rollout/update ablation",
    )
    ctx.invoke(
        metric,
        repo_path=repo_path,
        metric_name="loss",
        tags=(f"task-rollout-update-{sweep_suffix}",),
        param_filters=("n_steps=256", "update_epochs=4"),
        label_params=("seed",),
        output=output_dir / "loss_selected.png",
        csv_output=None,
        aggregate=True,
        moving_average=101,
        max_runs=0,
        title="Selected configuration loss (mean ± std)",
    )
    ctx.invoke(
        metric,
        repo_path=repo_path,
        metric_name="episode/return_mean",
        tags=(f"task-rollout-update-{sweep_suffix}",),
        param_filters=("n_steps=256", "update_epochs=4"),
        label_params=("seed",),
        output=output_dir / "episode_return_selected.png",
        csv_output=None,
        aggregate=True,
        moving_average=21,
        max_runs=0,
        title="Selected configuration training return (mean ± std)",
    )


if __name__ == "__main__":
    cli()

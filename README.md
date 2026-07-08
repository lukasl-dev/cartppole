# cartPPOle

PPO for `CartPole-v1` with Gymnasium, PyTorch, and Aim.

It provides:

- training and evaluation scripts
- PPO sweeps / ablations
- Aim logging
- static plotting from Aim runs
- report and presentation sources

## Setup

```bash
uv sync
```

## Quick start

```bash
uv run make train
uv run make evaluate
```

Default checkpoint:

```text
checkpoints/policy.pt
```

## Usage

### Train

```bash
uv run make train
```

Preset sizes:

```bash
uv run make train_xs
uv run make train_sm
uv run make train_md
uv run make train_lg
uv run make train_xl
uv run make train_xxl
```

Render while training:

```bash
uv run make train_visual
```

### Evaluate

```bash
uv run make evaluate
```

With args:

```bash
uv run make evaluate ARGS="--checkpoint-path checkpoints/policy.pt --n-episodes 60"
```

### Play

```bash
uv run make play
```

### Ablations

Quick run:

```bash
uv run make task_ablation_quick
```

Full 1M-step run:

```bash
uv run make task_ablation_large
```

This runs:

- clip range ablation
- GAE / Monte-Carlo ablation
- rollout/update ablation

### Plots

Plots are generated from Aim logs:

```bash
uv run make plot
```

Outputs go to `plots/`.

Custom metric example:

```bash
uv run python scripts/plot_results.py metric \
  --metric policy/entropy \
  --tag task-rollout-update-1000000 \
  --param n_steps=256 \
  --param update_epochs=4 \
  --aggregate \
  --output plots/entropy_selected.png
```

### Aim

```bash
uv run make aim
```


## Optional: Nix

```bash
nix develop --accept-flake-config
nix run .#check --accept-flake-config
nix build .#report .#presentation --accept-flake-config
```

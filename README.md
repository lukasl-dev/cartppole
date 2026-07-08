# cartPPOle

PPO implementation for `CartPole-v1` using Gymnasium and PyTorch.

## Development

Enter the project environment, then use the `Makefile` targets below.

### Train

```sh
make train
```

Smaller/larger training runs:

```sh
make train_xs
make train_sm
make train_md
make train_lg
make train_xl
make train_xxl
```

Train with rendering enabled:

```sh
make train_visual
```

The default checkpoint is saved to:

```text
checkpoints/policy.pt
```

### Play a checkpoint

```sh
make play
```

This loads `checkpoints/policy.pt` and runs one rendered episode.

### Aim UI

Training metrics are tracked with Aim. Start the UI with:

```sh
make aim
```

### Plot static figures

Aim is used as the experiment store, and `scripts/plot_results.py` exports
reproducible static plots from it:

```sh
make plot
```

This writes standard ablation and training-curve figures to `plots/`. Use
`python scripts/plot_results.py --help` for custom metrics, tags, and grouping.
The command expects an Aim repository from prior training/sweep runs; generate
one with `make task_ablation_large` or a smaller sweep first.

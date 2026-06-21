export PYTHONPATH=src

CARTPPOLE_SOURCES := $(shell find src/cartppole -type f -name '*.py')

# https://arxiv.org/abs/2109.08203
SWEEP_FLAGS := --seeds 0,1,2,3407 --clip-ranges 0.1,0.2,0.3 --learning-rates 0.0001,0.00025,0.0005 --value-coefs 0.0,0.25,0.5 --entropy-coefs 0.0,0.01 --normalise-advantages true,false --policy-losses clipped,unclipped --gae-lambdas 0.9,0.95,0.97 --advantage-estimators gae,mc
TASK_ABLATION_SEEDS ?= 0,1,2,3407
TASK_ABLATION_TIMESTEPS ?= 500000
TASK_ABLATION_EVAL_EPISODES ?= 60
TASK_ABLATION_SUCCESS_THRESHOLD ?= 475
TASK_ABLATION_COMMON_FLAGS := --seeds $(TASK_ABLATION_SEEDS) --total-timesteps $(TASK_ABLATION_TIMESTEPS) --n-eval-episodes $(TASK_ABLATION_EVAL_EPISODES) --success-threshold $(TASK_ABLATION_SUCCESS_THRESHOLD) --learning-rates 0.00025 --value-coefs 0.5 --entropy-coefs 0.01 --normalise-advantages true --policy-losses clipped
TASK_ABLATION_LARGE_TIMESTEPS ?= 1000000
TASK_ABLATION_CONFIRM_SEEDS ?= 0,1,2,3,4,5,6,7,8,9
TASK_ABLATION_CONFIRM_TIMESTEPS ?= 1000000
TASK_ABLATION_CONFIRM_FLAGS := --seeds $(TASK_ABLATION_CONFIRM_SEEDS) --total-timesteps $(TASK_ABLATION_CONFIRM_TIMESTEPS) --n-eval-episodes $(TASK_ABLATION_EVAL_EPISODES) --success-threshold $(TASK_ABLATION_SUCCESS_THRESHOLD) --learning-rates 0.00025 --value-coefs 0.5 --entropy-coefs 0.01 --normalise-advantages true --policy-losses clipped

.PHONY: dumb_game train train_visual train_xs train_sm train_md train_lg train_xl train_xxl sweep sweep_xs sweep_sm sweep_md sweep_lg sweep_xl sweep_xxl ablation ablation_xs ablation_sm ablation_md ablation_lg ablation_xl ablation_xxl task_ablation task_ablation_quick task_ablation_large task_ablation_advantage_confirm evaluate play docs docs/gen docs/open aim

dumb_game:
	python scripts/dumb_game.py

train:
	python src/cartppole/train.py

train_visual:
	python src/cartppole/train.py --render --n-envs 1

train_xs:
	python src/cartppole/train.py --total-timesteps 1024 --checkpoint-path checkpoints/policy_xs.pt

train_sm:
	python src/cartppole/train.py --total-timesteps 10000 --checkpoint-path checkpoints/policy_sm.pt

train_md:
	python src/cartppole/train.py --total-timesteps 100000 --checkpoint-path checkpoints/policy_md.pt

train_lg:
	python src/cartppole/train.py --total-timesteps 500000 --checkpoint-path checkpoints/policy_lg.pt

train_xl:
	python src/cartppole/train.py --total-timesteps 1000000 --checkpoint-path checkpoints/policy_xl.pt

train_xxl:
	python src/cartppole/train.py --total-timesteps 5000000 --checkpoint-path checkpoints/policy_xxl.pt

sweep:
	python src/cartppole/sweep.py $(SWEEP_FLAGS) $(ARGS)

sweep_xs:
	python src/cartppole/sweep.py $(SWEEP_FLAGS) --total-timesteps 1024 $(ARGS)

sweep_sm:
	python src/cartppole/sweep.py $(SWEEP_FLAGS) --total-timesteps 10000 $(ARGS)

sweep_md:
	python src/cartppole/sweep.py $(SWEEP_FLAGS) --total-timesteps 100000 $(ARGS)

sweep_lg:
	python src/cartppole/sweep.py $(SWEEP_FLAGS) --total-timesteps 500000 $(ARGS)

sweep_xl:
	python src/cartppole/sweep.py $(SWEEP_FLAGS) --total-timesteps 1000000 $(ARGS)

sweep_xxl:
	python src/cartppole/sweep.py $(SWEEP_FLAGS) --total-timesteps 5000000 $(ARGS)

ablation:
	python src/cartppole/ablation.py $(ARGS)

ablation_xs:
	python src/cartppole/ablation.py --total-timesteps 1024 $(ARGS)

ablation_sm:
	python src/cartppole/ablation.py --total-timesteps 10000 $(ARGS)

ablation_md:
	python src/cartppole/ablation.py --total-timesteps 100000 $(ARGS)

ablation_lg:
	python src/cartppole/ablation.py --total-timesteps 500000 $(ARGS)

ablation_xl:
	python src/cartppole/ablation.py --total-timesteps 1000000 $(ARGS)

ablation_xxl:
	python src/cartppole/ablation.py --total-timesteps 5000000 $(ARGS)

task_ablation:
	python src/cartppole/sweep.py $(TASK_ABLATION_COMMON_FLAGS) --name task-clip-range-$(TASK_ABLATION_TIMESTEPS) --checkpoint-path checkpoints/task_ablation/$(TASK_ABLATION_TIMESTEPS)/clip_range.pt --clip-ranges 0.1,0.2,0.3 --advantage-estimators gae --gae-lambdas 0.95 --rollout-update-ratios 128:4 $(ARGS)
	python src/cartppole/sweep.py $(TASK_ABLATION_COMMON_FLAGS) --name task-gae-lambda-$(TASK_ABLATION_TIMESTEPS) --checkpoint-path checkpoints/task_ablation/$(TASK_ABLATION_TIMESTEPS)/gae_lambda.pt --clip-ranges 0.2 --advantage-estimators gae,mc --gae-lambdas 0.9,0.95,0.97 --rollout-update-ratios 128:4 $(ARGS)
	python src/cartppole/sweep.py $(TASK_ABLATION_COMMON_FLAGS) --name task-rollout-update-$(TASK_ABLATION_TIMESTEPS) --checkpoint-path checkpoints/task_ablation/$(TASK_ABLATION_TIMESTEPS)/rollout_update.pt --clip-ranges 0.2 --advantage-estimators gae --gae-lambdas 0.95 --rollout-update-ratios 64:4,128:4,256:4,128:8 $(ARGS)

task_ablation_quick:
	$(MAKE) task_ablation TASK_ABLATION_TIMESTEPS=100000 TASK_ABLATION_EVAL_EPISODES=$(TASK_ABLATION_EVAL_EPISODES) TASK_ABLATION_SUCCESS_THRESHOLD=$(TASK_ABLATION_SUCCESS_THRESHOLD) ARGS="$(ARGS)"

task_ablation_large:
	$(MAKE) task_ablation TASK_ABLATION_TIMESTEPS=$(TASK_ABLATION_LARGE_TIMESTEPS) TASK_ABLATION_EVAL_EPISODES=$(TASK_ABLATION_EVAL_EPISODES) TASK_ABLATION_SUCCESS_THRESHOLD=$(TASK_ABLATION_SUCCESS_THRESHOLD) ARGS="$(ARGS)"

task_ablation_advantage_confirm:
	python src/cartppole/sweep.py $(TASK_ABLATION_CONFIRM_FLAGS) --name task-advantage-confirm-$(TASK_ABLATION_CONFIRM_TIMESTEPS) --checkpoint-path checkpoints/task_ablation/confirm-$(TASK_ABLATION_CONFIRM_TIMESTEPS)/advantage.pt --clip-ranges 0.2 --advantage-estimators gae,mc --gae-lambdas 0.9,0.95,0.97 --rollout-update-ratios 128:4 $(ARGS)

evaluate:
	python src/cartppole/evaluate.py $(ARGS)

play:
	python src/cartppole/play.py $(ARGS)

docs: docs/gen docs/open

docs/gen: docs/.stamp

docs/.stamp: $(CARTPPOLE_SOURCES) pyproject.toml uv.lock
	mkdir -p docs
	uv run pdoc --math -o docs/ cartppole
	touch $@

docs/open: docs/.stamp
	xdg-open docs/cartppole.html

aim:
	aim up

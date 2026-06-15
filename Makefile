export PYTHONPATH=src

CARTPPOLE_SOURCES := $(shell find src/cartppole -type f -name '*.py')

# https://arxiv.org/abs/2109.08203
SWEEP_FLAGS := --seeds 0,1,2,3407 --clip-ranges 0.1,0.2,0.3 --learning-rates 0.0001,0.00025,0.0005 --value-coefs 0.0,0.25,0.5 --entropy-coefs 0.0,0.01 --normalise-advantages true,false --policy-losses clipped,unclipped --gae-lambdas 0.9,0.95,0.97 --advantage-estimators gae,mc

.PHONY: dumb_game train train_visual train_xs train_sm train_md train_lg train_xl train_xxl sweep sweep_xs sweep_sm sweep_md sweep_lg sweep_xl sweep_xxl evaluate play docs docs/gen docs/open aim

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

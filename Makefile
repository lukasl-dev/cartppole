export PYTHONPATH=src

.PHONY: dumb_game train train_visual train_xs train_sm train_md train_lg train_xl train_xxl evaluate play aim

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

evaluate:
	python src/cartppole/evaluate.py $(ARGS)

play:
	python src/cartppole/play.py $(ARGS)

aim:
	aim up

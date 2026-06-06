export PYTHONPATH=src

.PHONY: dumb_game train train_visual train_xs train_sm train_md train_lg train_xl train_xxl play aim

dumb_game:
	python scripts/dumb_game.py

train:
	python src/cartppole/train.py

train_visual:
	python src/cartppole/train.py --render --n-envs 1

train_xs:
	python src/cartppole/train.py --total-timesteps 1024

train_sm:
	python src/cartppole/train.py --total-timesteps 10000

train_md:
	python src/cartppole/train.py --total-timesteps 100000

train_lg:
	python src/cartppole/train.py --total-timesteps 500000

train_xl:
	python src/cartppole/train.py --total-timesteps 1000000

train_xxl:
	python src/cartppole/train.py --total-timesteps 5000000

play:
	python src/cartppole/play.py

aim:
	aim up

from cartppole.environment import Environment

import numpy as np

if __name__ == "__main__":
    env = Environment(n_envs=1, render=True)
    _ = env.reset(seed=42)

    for _ in range(500):
        action = np.array([env.action_space.sample()])
        step = env.step(action)
        print(f"{step.obs=} {step.reward=} {step.terminated=}")
        if step.terminated:
            _ = env.reset()

    env.close()

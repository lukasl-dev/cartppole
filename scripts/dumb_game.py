from cartppole.environment import Environment

import numpy as np
from aim import Run

if __name__ == "__main__":
    run = Run()

    env = Environment(n_envs=1, render=True)
    _ = env.reset(seed=42)

    episode_reward = 0
    episode = 0

    for step_count in range(500):
        action = np.array([env.action_space.sample()])
        step = env.step(action)

        episode_reward += float(step.reward[0])
        run.track(episode_reward, name="reward", step=step_count)

        if step.terminated[0]:
            run.track(episode_reward, name="episode_return", step=episode)
            episode_reward = 0
            episode += 1
            _ = env.reset()

    env.close()

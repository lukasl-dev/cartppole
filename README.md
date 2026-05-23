# cartPPOle

## Background and Motivation

Reinforcement Learning (RL) studies how an agent can learn to make decisions by
interacting with an environment and maximising long-term reward. In
policy-gradient methods, the agent directly optimises a parameterised policy,
which is often more natural than learning a value function for problems with
stochastic, high-variance dynamics. Proximal Policy Optimisation (PPO) is a
widely-used on-policy algorithm that stabilises policy-gradient updates via a
clipped surrogate objective, achieving strong empirical performance with
relatively simple implementation.

## Problem Description

The primary goal is to train an agent with PPO to solve CartPole and evaluate
performance, stability, and key design choices.

### Environment and Setup

- Use the **CartPole-v1** environment (discrete actions).

### PPO Implementation

Implement PPO with the following components:

- An **actor** network producing a categorical action distribution and a critic
  network predicting state value.
- Advantage estimation (e.g., GAE)
- PPO clipped surrogate loss with clipping parameter $\varepsilon$
- Value-function loss and entropy bonus
- Mini-batch updates over collected rollouts (multiple epochs per rollout)

### Evaluation and Analysis

- **Primary metric:** average episode return (and success rate if you define
  a success threshold)
- **Stability:** include mean $\pm$ std across seeds
- **(Optional) Clip range ablation:** $\varepsilon \in \{ 0.1, 0.2, 0.3 \}$
- **(Optional) GAE ablation:** $\lambda \in \{ 0.9, 0.95, 0.97 \}$ to compare
  to Monte-Carlo returns
- **(Optional) Rollout/update ratio:** vary rollout length or number of epochs
  per rollout and discuss sample efficiency vs overfitting 

## Expected Deliverables

Students working in groups (ideally 3 members) are expected to:

- produce a project report,
- deliver a group presentation,
- submit well-documented source code (training + evaluation + plotting scripts)

## Suggested Resources and References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
   Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.
   <https://arxiv.org/pdf/1707.06347>
2. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An
   Introduction (2nd ed.).
   <http://incompleteideas.net/book/the-book-2nd.html>
3. OpenAI Gym / Gymnasium (CartPole environment).
   <https://www.gymlibrary.dev/>
4. OpenAI Spinning Up in Deep RL (PPO reference and explanations).
   <https://spinningup.openai.com/en/latest/algorithms/ppo.htm>
5. Stable-Baselines3 (reference PPO implementation).
   <https://github.com/DLR-RM/stable-baselines3>
6. CleanRL (single-file PPO implementation, educational reference).
   <https://github.com/vwxyzjn/cleanrl>

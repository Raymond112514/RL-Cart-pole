# Reinforcement Learning: Cart pole

In this project, both a `QLearningAgent` and `DeepQLearningAgent` are implemented and trained to interact and play with the OpenAI Gym CartPole environment. In the CartPole game, the goal is to balance a pole on top of a moving cart, navigating the system to prevent the pole from falling over. Players aim to achieve a stable equilibrium by strategically controlling the cart's movements, with the objective of maximizing the duration the pole remains upright to score higher points.

A agent can be created by instantiating either the `QLearningAgent` and `DeepQLearningAgent` class. Each agent instance contains a `choose` method which chooses the optimal policy based on the epsilon greedy strategy. The `Qlearning` and `DeepQlearning` files contain an implementation of the training loop, and the `utils.py` file contains implementations of functions used for graphing and computing moving averages.

## Q learning agent

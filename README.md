# Reinforcement Learning: Cart pole

In this project, both a `QLearningAgent` and `DeepQLearningAgent` are implemented and trained to interact and play with the OpenAI Gym CartPole environment. In the CartPole game, the goal is to balance a pole on top of a moving cart, navigating the system to prevent the pole from falling over. Players aim to achieve a stable equilibrium by strategically controlling the cart's movements, with the objective of maximizing the duration the pole remains upright to score higher points.

In the CartPole environment, the states represent the current configuration of the system, including the position and velocity of the cart as well as the angle and angular velocity of the pole. These state variables influence the agent's decisions and play a crucial role in the learning process.

An agent can be created by instantiating either the `QLearningAgent` and `DeepQLearningAgent` class. Each agent instance contains a `choose` method which chooses the optimal policy based on the epsilon greedy strategy. The `Qlearning` and `DeepQlearning` files contain an implementation of the training loop, and the `utils.py` file contains implementations of functions used for graphing and computing moving averages.

## Q learning agent

A Q learning agent creates a Q-value table for each of the discretized states (position, velocity, angle, angular velocity). The agent is trained with learning rate of $0.10$, discount of $0.90$, and epsilon of $0.75$. The learning rate $\eta$ and epsilon $\epsilon$ is decreased exponentially by the following scheme

\[
\begin{equation*}
  \begin{cases}
    \eta = \max(\eta * (1-10^{-8}), 0.01)\\
    \epsilon = \max(\epsilon * (1-10^{-6}), 0.01)\\
  \end{cases}
\end{equation*}
\]

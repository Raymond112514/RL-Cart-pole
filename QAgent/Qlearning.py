from QlearningAgent import QlearningAgent
import matplotlib.pyplot as plt
from utils import discretizeStates, plot, plotHist, validate
import numpy as np
import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    states = discretizeStates(30)
    agent = QlearningAgent(states, 2)
    n_episodes = 300
    scores = []

    for ep in range(1, n_episodes + 1):
        done = False
        state = env.reset()[0]
        score = 0
        while not done:
            action = agent.chooseAction(state)
            state_, reward, done, _, _ = env.step(action)
            agent.learn(state, action, reward, state_)
            score += reward
            state = state_
        scores.append(score)
        if ep % 100 == 0:
            print(f"Episode:{ep} with score: {score} and learning rate: {agent.lr} and epsilon: {agent.eps}")
            print(f"------------------------------------------------------------------------------------------------"
                  f"--------------------------------------------")
    print(f"The mean score is: {sum(scores) / len(scores)}")

    plot(scores, "Q-learning")
    plotHist(scores, "Q-learning")
    validate(agent)









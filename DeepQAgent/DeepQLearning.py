from DeepQLearningAgent import DeepQlearningAgent
from utils import plot, plotHist, validate
import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = DeepQlearningAgent(input_dim=4, n_actions=2)
    n_episodes = 300
    scores = []

    for ep in range(1, n_episodes + 1):
        done = False
        state = env.reset()[0]
        score = 0
        while not done:
            action = agent.chooseAction(state)
            state_, reward, done, _, _ = env.step(action)
            agent.store(state, action, reward, state_, done)
            score += reward
            state = state_
            loss = agent.learn()
        scores.append(score)
        if ep % 10 == 0:
            print(f"Episode:{ep} with score: {score} and learning rate: {agent.Q.optimizer.param_groups[0]['lr']}, "
                  f"and epsilon: {agent.eps} with loss: {loss}")
            print(f"------------------------------------------------------------------------------------------------"
                  f"--------------------------------------------")
    print(f"The mean score is: {sum(scores) / len(scores)}")

    plot(scores, "Deep-Q-learning")
    plotHist(scores, "Deep-Q-learning")
    validate(agent)


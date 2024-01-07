import numpy as np
import matplotlib.pyplot as plt
import imageio
import gym

def discretizeStates(n_partitions=10):
    position = np.linspace(-4.8, 4.8, n_partitions)
    velocity = np.linspace(-4, 4, n_partitions)
    angle = np.linspace(-0.418, 0.418, n_partitions)
    ang_velocity = np.linspace(-4, 4, n_partitions)
    return position, velocity, angle, ang_velocity

def plot(scores, title):
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Evolution of scores under " + title + " policy")
    plt.savefig("graphics/" + title + '_line_plot.png')
    plt.close()

def plotHist(scores, title):
    scores = np.clip(scores, None, 500)
    plt.hist(scores, bins=20)
    plt.xlim(0, 500)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Histogram of Scores")
    plt.savefig("graphics/" + title + '_histogram.png')
    plt.close()

def validate(agent, n_episodes=5):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    for ep in range(1, n_episodes + 1):
        done = False
        state = env.reset()[0]
        score = 0
        frames = []
        while not done:
            frame = env.render()
            action = agent.chooseAction(state, explore=False)
            state_, reward, done, _, _ = env.step(action)
            score += reward
            frames.append(frame)
            state = state_
        print(f"Validation Episode:{ep} with score: {score}")

        imageio.mimsave(f"graphics/episode_{ep}.gif", frames, duration=0.05)




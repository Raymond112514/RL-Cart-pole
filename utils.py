import numpy as np
import matplotlib.pyplot as plt
import imageio
import gym

maximum = 250

def discretizeStates(n_partitions=10):
    position = np.linspace(-4.8, 4.8, n_partitions)
    velocity = np.linspace(-4, 4, n_partitions)
    angle = np.linspace(-0.418, 0.418, n_partitions)
    ang_velocity = np.linspace(-4, 4, n_partitions)
    return position, velocity, angle, ang_velocity

def plot(scores, title, window_size=50):

    scores = preprocess(scores)
    movingAvg = moving_average(scores, window_size)
    plt.plot(scores)
    plt.plot(np.arange(window_size - 1, len(scores)), movingAvg, color='red')
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.ylim(0, maximum)
    plt.title("Evolution of scores under " + title + " policy")
    plt.savefig("graphics/" + title + '_line_plot.png')
    plt.close()

def preprocess(scores, maxVal=maximum):
    returnScores = []
    for score in scores:
        if score >= maxVal:
            returnScores.append(maxVal)
        else:
            returnScores.append(score)
    return returnScores

def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

def plotHist(scores, title):
    scores = np.clip(scores, None, maximum)
    plt.hist(scores, bins=20)
    plt.xlim(0, maximum + 50)
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




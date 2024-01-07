import random
import gym
import matplotlib.pyplot as plt

def randomPlay(episodes, visualize=False):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    scores = []
    for ep in range(1, episodes + 1):
        env.reset()
        done = False
        score = 0
        while not done:
            if visualize:
                frame = env.render()
                plt.imshow(frame)
                plt.axis('off')
                plt.show(block=False)
                plt.pause(0.01)
            action = random.choice([0, 1])
            n_state, reward, done, info, _ = env.step(action)
            score += reward
        if visualize:
            plt.close()
        if ep % 100 == 0:
            print(f"Episode: {ep}, score: {score}")
        scores.append(score)
    print(f"The mean score is: {sum(scores) / episodes}")

    plt.hist(scores, bins=30)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Histogram of scores")
    plt.savefig('random_histogram.png')
    plt.close()

randomPlay(20000, False)


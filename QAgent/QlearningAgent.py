import numpy as np

class QlearningAgent:

    """
    Constructor of Q-learning agent:
        n_actions: number of actions
        lr: learning rate
        eps: exploitation factor epsilon
        eps_dr: epsilon decrease rate (exponential decrease)
    """
    def __init__(self, states, n_actions, lr=0.10, eps=0.75, eps_dr=0.9995, discount=0.9):
        self.pos, self.vel, self.ang, self.ang_vel = states[0], states[1], states[2], states[3]
        self.n_states = len(self.pos) ** 4
        self.n_actions = n_actions
        self.lr = lr
        self.eps = eps
        self.eps_decay_rate = eps_dr
        self.discount = discount
        self.Q = {}
        self.initializeQValues()

    """
    InitializeQValues:
        Initializes the q-values as 0 for each (state, action) pair
    """
    def initializeQValues(self):
        for p in self.pos:
            for v in self.vel:
                for a in self.ang:
                    for av in self.ang_vel:
                        for action in range(self.n_actions):
                            self.Q[((p, v, a, av), action)] = 0.0

    """
    getDiscretizedState:
        Given an observation (position, velocity, angle, angular velocity), returns the discretized
        state
    """
    def getDiscretizedState(self, observation):
        p, v, a, av = observation
        p = self.pos[np.argmin(np.abs(np.array(self.pos) - p))]
        v = self.vel[np.argmin(np.abs(np.array(self.vel) - v))]
        a = self.ang[np.argmin(np.abs(np.array(self.ang) - a))]
        av = self.ang_vel[np.argmin(np.abs(np.array(self.ang_vel) - av))]
        return (p, v, a, av)

    """
    decreaseEps:
        Decrease epsilon using exponential decay with rate of eps_dr
    """
    def decreaseEps(self):
        self.eps = max(self.eps * (1 - 10 ** (-6)), 0.01)
        self.lr = max(self.lr * (1 - 10 ** (-8)), 0.01)

    """"""
    def chooseAction(self, state, explore=True):
        state = self.getDiscretizedState(state)
        if np.random.random() < self.eps and explore:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            action = np.argmax(np.array([self.Q[(state, a)] for a in range(self.n_actions)]))
        return action

    def learn(self, state, action, reward, state_):
        state = self.getDiscretizedState(state)
        state_ = self.getDiscretizedState(state_)
        action_ = np.argmax(np.array([self.Q[(state_, a)] for a in range(self.n_actions)]))
        self.Q[(state, action)] += self.lr * (reward + self.discount * self.Q[(state_, action_)] - self.Q[(state, action)])
        self.decreaseEps()
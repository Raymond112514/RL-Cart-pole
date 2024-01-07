import numpy as np
from DeepQNetwork import DeepQNetwork
import torch as T

class DeepQlearningAgent:
    def __init__(self, input_dim, n_actions, batch_size=64, mem_size=1000, lr=0.01, eps=0.10, eps_dr=0.9995, discount=0.99):
        self.n_actions = n_actions
        self.eps = eps
        self.eps_decay_rate = eps_dr
        self.discount = discount
        self.Q = DeepQNetwork(input_dim=4, n_actions=2, lr=lr)
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mem_counter = 0
        self.state_mem = np.zeros((self.mem_size, input_dim), dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self._state_mem = np.zeros((self.mem_size, input_dim), dtype=np.float32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.int32)

    def decreaseEps(self):
        self.eps = max(self.eps * (1 - 10 ** (-4)), 10 ** -5)

    def chooseAction(self, state, explore=True):
        if np.random.random() < self.eps and explore:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            state = T.tensor(state, dtype=T.float).to(self.Q.device)
            action = T.argmax(self.Q.forward(state)).item()
        return action

    def store(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_mem[index] = state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self._state_mem[index] = state_
        self.terminal_mem[index] = 0 if done else 1
        self.mem_counter += 1

    def learn(self):
        if self.mem_counter < self.mem_size:
            return
        self.Q.optimizer.zero_grad()
        mem_size = min(self.mem_size, self.mem_counter)
        batch = np.random.choice(mem_size, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state = T.tensor(self.state_mem[batch]).to(self.Q.device)
        action = self.action_mem[batch]
        state_ = T.tensor(self._state_mem[batch]).to(self.Q.device)
        reward = T.tensor(self.reward_mem[batch]).to(self.Q.device)
        done = T.tensor(self.terminal_mem[batch]).to(self.Q.device)
        q_pred = self.Q.forward(state)[batch_index, action]
        q_next = self.Q.forward(state_)
        q_target = reward + self.discount * T.max(q_next, dim=1)[0] * done
        # print(q_target)
        # print(q_pred)
        loss = self.Q.loss(q_pred, q_target).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.Q.scheduler.step()
        self.decreaseEps()
        return loss




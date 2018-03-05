import numpy as np
import tensorflow as tf

from collections import deque
import random
from dqn import DQN

def avg(a):
    return sum(a) / len(a)

class Training:
    def __init__(self, env):
        self.env = env
        self.dqn = DQN(env.observation_space.shape, env.action_space.n)

        self.n_greedy_runs = 0


        self.memory = deque(maxlen=100)
        # self.worst_memories = deque(maxlen=5)
        self.best_memories = deque(maxlen=5)
        self.random_explore = True

    def explore(self, n):
        for i in range(n):
            exp = self.test_explore()


            if not self.random_explore:
                if len(self.best_memories) == 0:
                    self.best_memories.append(exp)
                elif exp.total_rew > avg([e.total_rew for e in self.best_memories]):
                    self.best_memories.append(exp)

                self.memory.append(exp)


    def avg_max_scores(self, n_tries):

            scores = []
            for j in range(n_tries):
                scores.append(self.test_greedy())

            return avg(scores), max(scores)

    def train(self, n_epochs):
        if not self.random_explore:
            for i in range(n_epochs):
                lst = []
                lst.extend(self.best_memories)
                lst.extend(random.sample(self.memory,min(len(self.memory), 25)))
                random.shuffle(lst)
                for exp in lst:
                    self.dqn.train_on_exp(exp)




    def evaluate(self, dots):
        return self.dqn.evaluate(dots)

    def test_greedy(self, render = False):
        obs = self.env.reset()
        done = False
        total_rew = 0
        while not done:
            action = self.dqn.greedy_choice(obs)
            if render:
                self.env.render()
            obs, reward, done, _ = self.env.step(action)
            total_rew += reward

        summary = tf.Summary()
        summary.value.add(tag='reward per episode', simple_value=total_rew)
        self.n_greedy_runs+=1
        self.dqn.writter.add_summary(summary, self.n_greedy_runs)


        return total_rew


    def test_explore(self):
        obs = self.env.reset()
        done = False
        total_rew = 0
        s = []
        a = []
        r = []
        sn = []
        while not done:
            if self.random_explore:
                action = self.env.action_space.sample()
            else:
                action = self.dqn.exploratory_choice(obs)
            s.append(obs)
            # print(action.shape)
            obs, reward, done, _ = self.env.step(action)
            a.append(action)
            r.append(reward)
            sn.append(obs)

            total_rew += reward
            # print(obs)

        if total_rew >= -20:
            self.random_explore = False

        return Experience(np.vstack(s), np.vstack(a), np.vstack(r), np.vstack(sn), total_rew)


class Experience:
    def __init__(self, s, a, r, sn, total_rew):
        self.values = (s, a, r, sn)
        self.total_rew = total_rew


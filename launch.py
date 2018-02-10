from dqn import DQN
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from my_env import envs

for env in envs:
    # if 'Box' in str(env.action_space):
    #     dqn = DQN(env.observation_space.shape, env.action_space.shape[0])
    # else:

    dqn = DQN(env.observation_space.shape, env.action_space.n)

    n_greedy_runs = 0

    def test_greedy(render = False):
        obs = env.reset()
        done = False
        total_rew = 0
        while not done:
            action = dqn.greedy_choice(obs)
            if render:
                env.render()
            obs, reward, done, _ = env.step(action)
            total_rew += reward

        summary = tf.Summary()
        summary.value.add(tag='reward per episode', simple_value=total_rew)
        global n_greedy_runs
        n_greedy_runs+=1
        dqn.writter.add_summary(summary, n_greedy_runs)


        return total_rew

    def test_explore():
        obs = env.reset()
        done = False
        total_rew = 0
        s = []
        a = []
        r = []
        sn = []
        while not done:
            action = dqn.exploratory_choice(obs)
            s.append(obs)
            # print(action.shape)
            obs, reward, done, _ = env.step(action)
            a.append(action)
            r.append(reward)
            sn.append(obs)

            total_rew += reward
            # print(obs)
        # r[-1] = total_rew - 200

        return np.vstack(s), np.vstack(a), np.vstack(r), np.vstack(sn), total_rew

    running_avg = deque(maxlen=30)
    batches = deque(maxlen=25)

    def train():
        for i in range(1000):

            s, a, r, sn, _ = test_explore()
            batches.append((s, a, r, sn))

            for batch in batches:
                s, a, r, sn = batch
                dqn.train_on_exp_batch(s, a, r, sn)

            if i % 50 == 0:
                whee_count = 0
                for j in range(10):
                    tg = test_greedy()
                    # if tg == 200:
                    #     whee_count += 1
                    #     if whee_count > 7:
                    #         print("wheee!")
                    #         return
                    running_avg.append(tg)
                print(sum(running_avg) / len(running_avg))

    test_greedy(True)
    train()
    test_greedy(True)
from dqn import DQN
import gym

env = gym.make('CartPole-v0')

dqn = DQN(env.observation_space.shape, env.action_space.n)

def test_greedy():
    obs = env.reset()
    done = False
    total_rew = 0
    while not done:
        action = dqn.greedy_choice(obs)
        # print(action.shape)
        obs, reward, done, _ = env.step(action)
        total_rew += reward
        # print(obs)

    print("gained total of {} points".format(total_rew))

def test_explore():
    obs = env.reset()
    done = False
    total_rew = 0
    while not done:
        action = dqn.exploratory_choice(obs)
        # print(action.shape)
        obs, reward, done, _ = env.step(action)
        total_rew += reward
        # print(obs)

    print("gained total of {} points".format(total_rew))


test_explore()
test_greedy()

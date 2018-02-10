import gym


# env_names = ['Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0']
env_names = ['MountainCar-v0']


envs = {1}
envs.pop()
for name in env_names:
    envs.add(gym.make(name))

for env in envs:
    print(env)
    print(env.action_space)
    print(env.observation_space)

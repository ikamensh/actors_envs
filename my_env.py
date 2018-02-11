import gym
import numpy as np


# env_names = ['Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0']
# env_names = ['MountainCar-v0']
# env_names = ['CartPole-v0']
#
#
# envs = {1}
# envs.pop()
# for name in env_names:
#     envs.add(gym.make(name))
#
# for env in envs:
env = gym.make('CartPole-v0')

print(env)
print(env.action_space)
print(env.observation_space)

arx = np.linspace(env.observation_space.low[0], env.observation_space.high[0])
ary = np.linspace(env.observation_space.low[2], env.observation_space.high[2])

def get_grid(dx, dphi):
    dots = np.ndarray([len(arx) * len(ary), 4], dtype=np.float32)
    for i in range(len(arx)):
        for j in range(len(ary)):
            dots[i * len(arx) + j] = [arx[i], dx, ary[j], dphi]

    return dots

grid00 = get_grid(0,0)
grid10 = get_grid(0.3,0)
grid01 = get_grid(0,0.3)
gridn10 = get_grid(-0.3,0)
gridn01 = get_grid(0,-0.3)

grids = {}
grids['0_0'] = grid00
grids['1_0'] = grid01
grids['0_1'] = grid10
grids['n1_0'] = gridn01
grids['n0_1'] = gridn10



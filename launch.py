from my_env import env, grids
from training import Training
from utils import draw_2darray


# for env in envs:
    # if 'Box' in str(env.action_space):
    #     dqn = DQN(env.observation_space.shape, env.action_space.shape[0])
    # else:
for trial in range(10):
    t = Training(env)
    print('-=-°'*15 + "TRIAL #{}".format(trial) + '-=-°'*15)

    max_sustained = -1
    max_ever = -1
    for i in range(10):
        # t.test_greedy(True)


        for j in range(25):
            t.explore(5)
            t.train(5)

        a, m = t.avg_max_scores(10)
        if a > max_sustained:
            max_sustained = a
        if m > max_ever:
            max_ever = m
        print("scores: avg {} ; max {}".format(a,m))
        # t.test_greedy(True)
        # b = t.evaluate(dots)
    print('-=-°' * 15 + "ACIEVEMENTS: #{} / {}".format(max_sustained, max_ever) + '-=-°' * 15)



def evaluate():
    for name, grid in grids.items():
        a = t.evaluate(grid)
        a_1 = a[:,0].reshape(50,50)
        a_2 = a[:,1].reshape(50,50)
        draw_2darray(a_1, "plots/{}/a1/{}".format(name, i))
        draw_2darray(a_2, "plots/{}/a2/{}".format(name, i))
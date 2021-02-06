# import sys
# sys.path.append('./gym')

import gym
import numpy as np
from src.collector.collect import cartpole_collect


class runner:
    def __init__(self, num_times = 10):
        self.env = gym.make('CartPole-v0')
        self.ABs = []
        self.SAs = []
        self.Ss = []
        self.num_times = num_times

    def run_once(self):
        collector = cartpole_collect(self.env)
        state = self.env.reset()
        collector.add_state(state)

        for i in range(1000):
            action = self.env.action_space.sample()
            state, reward, done, info = self.env.step(action)

            collector.add_state(state)
            collector.add_action(action)

            if done:
                break

        collector.get_derivatives()

        return collector

    def run_times(self):
        for i in range(self.num_times):
            c = self.run_once()
            self.ABs += c.ABs
            self.SAs += c.SAs
            self.Ss += c.states[1:]

    def save_np(self, path='./cartpole_dataset'):
        dataset = np.array([{'ABs':self.ABs, 'SAs':self.SAs, 'Ss':self.Ss}])
        np.save(path, dataset)


    def show(self):
        print(self.store_ABs)
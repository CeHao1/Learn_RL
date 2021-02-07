# import sys
# sys.path.append('./gym')

import gym
import numpy as np
from tqdm import tqdm
from src.collector.collect_param import cartpole_collect_param
from src.collector.collect import  cartpole_collect


class runner:
    def __init__(self):
        self.env = gym.make('CartPole-v2')
        self.ABs = []
        self.SAs = []
        self.Ss = []
        self.dx = []
        self.ds = []
        self.params = []

    def run_once(self):
        collector = cartpole_collect_param(self.env)
        state = self.env.reset()
        collector.add_state(state)

        for i in range(1000):
            action = self.env.action_space.sample()
            state, reward, done, params = self.env.step(action)

            collector.add_state(state)
            collector.add_action(action)
            collector.add_params(params)
            if done:
                break

        collector.get_derivatives()

        return collector

    def run_times(self, num_times = 10):
        for i in tqdm(range(num_times)):
            c = self.run_once()
            self.ABs += c.ABs
            self.SAs += c.SAs
            self.Ss += c.states[1:]
            self.dx += c.x_dots
            self.ds += c.s_dots
            self.params += c.params

    def save(self, path='./cartpole_dataset'):
        dataset = np.array([{'ABs':self.ABs, 
                            'SAs':self.SAs, 
                            'Ss':self.Ss, 
                            'dx':self.dx,
                            'ds':self.ds,
                            'params':self.params}])
        np.save(path, dataset)


    def show(self):
        print(self.store_ABs)

    def clear(self):
        self.ABs = []
        self.SAs = []
        self.Ss = []
        self.dx = []
        self.ds = []
        self.params = []
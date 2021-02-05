import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import matplotlib.pyplot as plt

class DQN_frame:

    def __init__(self):
        self.num_iterations = 5000 # @param {type:"integer"}

        self.initial_collect_steps = 100  # @param {type:"integer"} 
        self.collect_steps_per_iteration = 1  # @param {type:"integer"}
        self.replay_buffer_max_length = 100000  # @param {type:"integer"}

        self.batch_size = 64  # @param {type:"integer"}
        self.learning_rate = 1e-3  # @param {type:"number"}
        self.log_interval = 200  # @param {type:"integer"}

        self.num_eval_episodes = 10  # @param {type:"integer"}
        self.eval_interval = 1000  # @param {type:"integer"}

        self.env_name = 'CartPole-v0'
        self.fc_layer_params = (100,)


    def build(self):

        # build environment
        self.train_py_env = suite_gym.load(self.env_name)
        self.eval_py_env = suite_gym.load(self.env_name)

        # we can chagne cartpole parameters here

        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)

        # build agent
        q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=self.fc_layer_params)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)

        self.agent.initialize()


        # build policy
        self.random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(),self.train_env.action_spec())

        # build replay buffer

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_max_length)

        # build collect
        self.collect_data(self.train_env, self.random_policy, self.replay_buffer, self.initial_collect_steps)

        # build dataset
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, 
            sample_batch_size=self.batch_size, 
            num_steps=2).prefetch(3)

        self.iterator = iter(self.dataset)

    def train(self):
        self.agent.train = common.function(self.agent.train)

        self.agent.train_step_counter.assign(0)

        avg_return = self.compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
        self.returns = [avg_return]

        for _ in range(self.num_iterations):
            self.collect_data(self.train_env, self.agent.collect_policy, self.replay_buffer, self.collect_steps_per_iteration)
            experience, unused_info = next(self.iterator)
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()
            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                avg_return = self.compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                self.returns.append(avg_return)


    def compute_avg_return(self, environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def collect_step(self, environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_data(self, env, policy, buffer, steps):
        for _ in range(steps):
            self.collect_step(env, policy, buffer)


    
    def plot(self):
        iterations = range(0, self.num_iterations + 1, self.eval_interval)
        plt.plot(iterations, self.returns)
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        plt.ylim(top=250)
        plt.show()
import numpy as np
import pygame
import math
import time
from load import *
from pygame.locals import *
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()

# 摆脱Gym提供的仿真环境，自己创建CartPole仿真环境
class CartPoleEnv:
    def __init__(self):
        self.actions = [0,1]
        self.state = np.random.uniform(-0.05, 0.05, size=(4,))
        self.steps_beyond_done = 0
        self.viewer = None
        # 设置帧率
        self.FPSCLOCK = pygame.time.Clock()
        self.screen_size = [400,300]
        self.cart_x = 200
        self.cart_y = 200
        self.theta = -1.5
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = (self.mass_cart + self.mass_pole)
        self.length = 0.5
        self.pole_mess_length = (self.mass_pole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02
    
    # 随机初始化
    def reset(self):
        n = np.random.randint(1, 1000, 1) 
        np.random.seed(n)
        self.state = np.random.uniform(-0.05, 0.05, size=(4,))
        self.steps_beyond_done = 0
        return np.array(self.state)
    
    def step(self, action):
        state = self.statex, x_dot, theta, theta_dot = stateforce = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        # 动力学方程
        temp = (force+self.pole_mess_length * theta_dot * theta_dot * sintheta)/self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp)/(self.length * (4.0/3.0-self.mass_pole*costheta*costheta/self.total_mass))
        xacc = temp - self.pole_mess_length * thetaacc * costheta / self.total_mass
        x = x+self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta +self.tau * theta_dot
        theta_dot = theta_dot * self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        # 判断是否结束
        done = x<-self.x_threhold or x > self.x_threhold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        done = bool(done)
        # 设置回报
        if not done:
            reward = 1.0
            self.steps_beyond_done = self.steps_beyond_done + 1
    
    def gameover(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
    
    def render(self):
        screen_width = self.screen_size[0]
        screen_height = self.screen_size[1]
        world_width = self.x_threhold * 2
        scale = screen_width/world_width
        state = self.state
        self.cart_x = 200 + scale*state[0]
        self.cart_y = 200
        self.theta = state[2]
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode
            (self.screen_size,0,32)
            self.background = load_background()
            self.pole = load_pole()
            self.viewer.blit(self.background,(0,0))
            self.viewer.blit(self.pole, (195,80))
            pygame.display.update()
        self.viewer.blit(self.background,(0,0))
        pygame.draw.line(self.viewer, (0,0,0),(0,200),(400,200))
        pygame.draw.rect(self.viewer,(250,0,0),(self.cart_x-20, self.cart_y-15,40,30))
        pole1=pygame.transform.rotate(self.pole, -self.theta*180/math.pi)
        if self.theta >0:
            pole1_x = self.cart_x-5*math.cos(self.theta)
            pole1_y = self.cart_y-80*math.cos(self.theta)-5*math.sin(self.theta)
        else:
            pole1_x = cart_x+80*math.sin(self.theta)-5*math.cos(self.theta)
            pole1_y = cart_y-80*math.cos(self.theta)+5*math.sin(self.theta)
        self.viewer.blit(pole1, (pole1_x, pole1_y))
        pygame.display.update()
        self.gameover()
        self.FPSCLOCK.tick(30)
            

class Sample():
    def __init__(self, env, policy_net):
        self.env = env
        self.policy_net = policy_net
        self.gamma = 0.98

    def sample_episodes(self, num_episodes):
        batch_obs = []
        batch_actions = []
        batch_rs = []
        for i in range(num_episodes):
            observation = self.env.reset()
            reward_episode = []
            while True:
                # if render: self.env.render()
                self.env.render()
                # 根据策略网络产生一个动作
                state = np.reshape(observation, [1,4])
                action = self.policy_net.choose_action(state)
                observation_, reward, done, info = self.env.step(action)
                batch_obs.append(observation)
                batch_actions.append(action)
                reward_episode.append(reward)
                if done:
                    reward_sum = 0
                    discounted_sum_reward = np.zeros_like(reward_episode)
                    for t in reversed(range(0, len(reward_episode))):
                        reward_sum = reward_sum * self.gamma + reward_episode[t]
                        discounted_sum_reward[t]=reward_sum
                    discounted_sum_reward -= np.mean(discounted_sum_reward)
                    discounted_sum_reward/= np.std(discounted_sum_reward)
                    for t in range(len(reward_episode)):
                        batch_rs.append(discounted_sum_reward[t])
                    break
                observation = observation_
        # 存储观测和回报
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.policy_net.n_features])
        batch_actions = np.reshape(batch_actions, [len(batch_actions),])
        batch_rs = np.reshape(batch_rs, [len(batch_rs),])
        return batch_obs, batch_actions, batch_rs
# 定义策略网络
class Policy_Net():
    def __init__(self, env, model_file=None):
        self.learning_rate = 0.01
        self.n_features = env.observation_space.shape[0] # 输入特征维度
        self.n_actions = env.action_space.n              # 输出动作空间维度
        self.obs = tf.placeholder(tf.float32, shape=[None, self.n_features])

        
        self.f1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.nn.relu, \
                                  kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.1),\
                                  bias_initializer=tf.constant_initializer(0.1))
        self.all_act = tf.layers.dense(inputs=self.f1, units=self.n_actions, activation=None,\
                                       kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.1),\
                                       bias_initializer=tf.constant_initializer(0.1))
        self.all_act_prob = tf.nn.softmax(self.all_act)
        self.current_act = tf.placeholder(tf.int32, [None,])    # 监督标签
        self.current_reward = tf.placeholder(tf.float32, [None,])


        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act, labels=self.current_act)
        self.loss = tf.reduce_mean(neg_log_prob*self.current_reward)    # 损失函数
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)  # 优化器
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
    # 定义贪婪网络
    def greedy_action(self, state):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.obs:state})
        action = np.argmax(prob_weights, 1)
        # print("greedy action:", action)
        return action[0]
    # 定义训练
    def train_step(self, state_batch, label_batch, reward_batch):
        loss,_ = self.sess.run([self.loss, self.train_op], feed_dict={self.obs:state_batch, \
                                                                      self.current_act:label_batch, \
                                                                      self.current_reward:reward_batch})
        return loss
    
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)
    # 依照概率选择动作
    def choose_action(self, state):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.obs:state})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        # print("action:", action)
        return action
    
def policy_train(env, brain, sample, training_num):
    reward_sum = 0
    reward_sum_line = []
    training_time = []
    for i in range(training_num):
        temp = 0
        training_time.append(i)
        train_obs, train_actions, train_rs = sample.sample_episodes(10)
        loss = brain.train_step(train_obs, train_actions, train_rs)
        print("current loss is %f" % loss)
        if i == 0:
            reward_sum = policy_test(env, brain, False, 1)
        else:
            reward_sum = 0.9 * reward_sum + 0.1 * policy_test(env, brain, False, 1)
        print(policy_test(env, brain, False, 1))
        reward_sum_line.append(reward_sum)
        print("training episodes is %d, trained reward_sum is %f" % (i, reward_sum))
        if reward_sum > 199:
            break
    brain.save_model('./current_best_pg_cartpole')
    plt.plot(training_time, reward_sum_line)
    plt.xlabel("training number")
    plt.ylabel("score")
    plt.show()

def policy_test(env, policy, render, test_num):
    for i in range(test_num):
        observation = env.reset()
        reward_sum = 0
        while True:
            if render: env.render()
            # from IPython import embed; embed(); exit();
            state = np.reshape(observation, [1,4])
            action = policy.greedy_action(state)
            observation_, reward, done, info = env.step(action)
            reward_sum += reward
            if done:
                break
            observation = observation_
    return reward_sum


if __name__ == '__main__':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.unwrapped
    env.seed(1)
    # env=CartPoleEnv()
    brain = Policy_Net(env, './current_best_pg_cartpole')  # 下载最好的模型
    # brain = Policy_Net(env)         # 策略网络实例化
    sampler = Sample(env, brain)    # 采样函数实例化
    training_num = 150              # 训练次数
    policy_train(env, brain, sampler, training_num)
    reward_sum = policy_test(env, brain, True, 10)
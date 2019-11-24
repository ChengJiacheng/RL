#!/usr/bin/env python
# coding: utf-8


# DQN, DDQN

get_ipython().system('nvidia-smi')


import warnings ; warnings.filterwarnings('ignore')
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from IPython.display import display
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from itertools import cycle, count
from textwrap import wrap


import matplotlib
import subprocess
import os.path
import tempfile
import random
import base64
import pprint
import time
import json
import sys
import gym
import io
import os

from gym import wrappers
from subprocess import check_output
from IPython.display import HTML

LEAVE_PRINT_EVERY_N_SECS = 20
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('.', 'results')
#SEEDS = (12, 34, 56, 78, 90)
SEEDS = (12, 34)

get_ipython().run_line_magic('matplotlib', 'auto')


plt.style.use('fivethirtyeight')
params = {
    'figure.figsize': (15, 8),
    'font.size': 24,
    'legend.fontsize': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
pylab.rcParams.update(params)
np.set_printoptions(suppress=True)



torch.cuda.is_available()




def get_make_env_fn(**kargs):
    def make_env_fn(env_name, seed=None, unwrapped=False, 
                    monitor_mode=None, addon_wrappers=None):
        mdir = tempfile.mkdtemp()
        env = gym.make(env_name)
        if seed is not None: env.seed(seed)
        env = env.unwrapped if unwrapped else env
        env = wrappers.Monitor(
            env, mdir, force=True, mode=monitor_mode) if monitor_mode else env
        if addon_wrappers:
            for wrapper in addon_wrappers:
                env = wrapper(env)
        return env
    return make_env_fn, kargs



def get_videos_html(env_videos, title, max_n_videos=5):
    videos = np.array(env_videos)
    if len(videos) == 0:
        return
    
    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1,]
    videos = videos[idxs,...]

    strm = '<h2>{}<h2>'.format(title)
    for video_path, meta_path in videos:
        video = io.open(video_path, 'r+b').read()
        encoded = base64.b64encode(video)

        with open(meta_path) as data_file:    
            meta = json.load(data_file)

        html_tag = """
        <h3>{0}<h3/>
        <video width="960" height="540" controls>
            <source src="data:video/mp4;base64,{1}" type="video/mp4" />
        </video>"""
        strm += html_tag.format('Episode ' + str(meta['episode_id']), encoded.decode('ascii'))
    return strm




def get_gif_html(env_videos, title, max_n_videos=5):
    videos = np.array(env_videos)
    if len(videos) == 0:
        return
    
    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1,]
    videos = videos[idxs,...]

    strm = '<h2>{}<h2>'.format(title)
    for video_path, meta_path in videos:
        basename = os.path.splitext(video_path)[0]
        gif_path = basename + '.gif'
        if not os.path.exists(gif_path):
            ps = subprocess.Popen(
                ('ffmpeg', 
                 '-i', video_path, 
                 '-r', '10', 
                 '-f', 'image2pipe', 
                 '-vcodec', 'ppm', 
                 '-'), 
                stdout=subprocess.PIPE)
            output = subprocess.check_output(
                ('convert', 
                 '-delay', '5', 
                 '-loop', '0', 
                 '-', gif_path), 
                stdin=ps.stdout)
            ps.wait()

        gif = io.open(gif_path, 'r+b').read()
        encoded = base64.b64encode(gif)
            
        with open(meta_path) as data_file:    
            meta = json.load(data_file)

        html_tag = """
        <h3>{0}<h3/>
        <img src="data:image/gif;base64,{1}" />"""
        strm += html_tag.format('Episode ' + str(meta['episode_id']), encoded.decode('ascii'))
    return strm


# # DQN


class FCQ(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super(FCQ, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, 
                                     hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(
                hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(
            hidden_dims[-1], output_dim)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             device=self.device, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x
    
    def numpy_float_to_device(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable
    
    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals




class GreedyStrategy():
    def __init__(self):
        self.exploratory_action_taken = False

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)




class EGreedyStrategy():
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.exploratory_action_taken = None

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else: 
            action = np.random.randint(len(q_values))

        self.exploratory_action_taken = action != np.argmax(q_values)
        return action



plt.figure()
s = EGreedyStrategy()
plt.plot([s.epsilon for _ in range(50000)])
plt.title('e-Greedy epsilon')
plt.xticks(rotation=45)
plt.show()




class EGreedyLinearStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.t = 0
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = decay_steps
        self.exploratory_action_taken = None
        
    def _epsilon_update(self):
        epsilon = 1 - self.t / self.decay_steps
        epsilon = (self.init_epsilon - self.min_epsilon) * epsilon + self.min_epsilon
        epsilon = np.clip(epsilon, self.min_epsilon, self.init_epsilon)
        self.t += 1
        return epsilon

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else: 
            action = np.random.randint(len(q_values))

        self.epsilon = self._epsilon_update()
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action



plt.figure()
s = EGreedyLinearStrategy()
plt.plot([s._epsilon_update() for _ in range(50000)])
plt.title('e-Greedy Linear epsilon')
plt.xticks(rotation=45)
plt.show()




class EGreedyExpStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).detach().cpu().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))

        self._epsilon_update()
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action



plt.figure()
s = EGreedyExpStrategy()
plt.plot([s._epsilon_update() for _ in range(50000)])
plt.title('e-Greedy Exponentially epsilon')
plt.xticks(rotation=45)
plt.show()




class SoftMaxStrategy():
    def __init__(self, 
                 init_temp=1.0, 
                 min_temp=0.3, 
                 exploration_ratio=0.8, 
                 max_steps=25000):
        self.t = 0
        self.init_temp = init_temp
        self.exploration_ratio = exploration_ratio
        self.min_temp = min_temp
        self.max_steps = max_steps
        self.exploratory_action_taken = None
        
    def _update_temp(self):
        temp = 1 - self.t / (self.max_steps * self.exploration_ratio)
        temp = (self.init_temp - self.min_temp) * temp + self.min_temp
        temp = np.clip(temp, self.min_temp, self.init_temp)
        self.t += 1
        return temp

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        temp = self._update_temp()

        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            scaled_qs = q_values/temp
            norm_qs = scaled_qs - scaled_qs.max()            
            e = np.exp(norm_qs)
            probs = e / np.sum(e)
            assert np.isclose(probs.sum(), 1.0)

        action = np.random.choice(np.arange(len(probs)), size=1, p=probs)[0]
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action



plt.figure()
s = SoftMaxStrategy()
plt.plot([s._update_temp() for _ in range(50000)])
plt.title('SoftMax Linear temperature')
plt.xticks(rotation=45)
plt.show()



class ReplayBuffer():
    def __init__(self, 
                 max_size=10000, 
                 batch_size=64):
        self.ss_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ds_mem = np.empty(shape=(max_size), dtype=np.ndarray)

        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    
    def store(self, sample):
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = d
        
        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(self.size, batch_size, replace=False)
        experiences = np.vstack(self.ss_mem[idxs]), np.vstack(self.as_mem[idxs]), np.vstack(self.rs_mem[idxs]), np.vstack(self.ps_mem[idxs]), np.vstack(self.ds_mem[idxs])
        return experiences

    def __len__(self):
        return self.size




class DQN():
    def __init__(self, 
                 reply_buffer_fn, 
                 value_model_fn, 
                 value_optimizer_fn, 
                 value_optimizer_lr,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps):
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)
        
        max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = self.online_model(states).gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, _ = env.step(action)
        past_limit_enforced = hasattr(env, '_past_limit') and env._past_limit()
        is_failure = is_terminal and not past_limit_enforced

        experience = (state, action, reward, new_state, float(is_failure))
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)
        return new_state, is_terminal
    
    def update_network(self):
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target.data.copy_(online.data)

    def train(self, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)
    
        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        self.episode_exploration = []
        
        self.target_model = self.value_model_fn(nS, nA)
        self.online_model = self.value_model_fn(nS, nA)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = training_strategy_fn()
        self.evaluation_strategy = evaluation_strategy_fn() 
                    
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()
            
            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                state, is_terminal = self.interaction_step(state, env)
                
                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)
                
                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_network()
                
                if is_terminal:
                    break
            
            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_model, env)
            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)
            
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:])/np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)
            
            wallclock_elapsed = time.time() - training_start
            result[episode-1] = total_step, mean_100_reward, mean_100_eval_score, training_time, wallclock_elapsed


            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward
            elapsed_str = time.strftime("%M:%S", time.gmtime(time.time() - training_start))
            
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode-1, total_step, mean_10_reward, std_10_reward, 
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            
            if (episode-1)%25 == 0 or training_is_over:
                print(episode, debug_message, end='\n')

            
#            if reached_debug_time or training_is_over:
#                print(ERASE_LINE + debug_message, flush=True)
#                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break
                
        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
        env.close() ; del env
        return result, final_eval_score, training_time, wallclock_time
    
    def evaluate(self, eval_policy_model, eval_env, n_episodes=5):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(self.online_model, s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)
    
    def demo(self, title='Trained {} Agent', n_episodes=10, max_n_videos=3):          
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation')
        self.evaluate(self.online_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)



dqn_results = []
dqn_agents, best_dqn_agent_key, best_eval_score = {}, None, float('-inf')

environment_settings = {
    'env_name': 'CartPole-v1',
    'gamma': 1,
    'max_minutes': 20,
    'max_episodes': 10000,
    'goal_mean_100_reward': 350
}

value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512, 128))
#value_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
value_optimizer_lr = 0.0005

# training_strategy_fn = lambda: EGreedyStrategy(epsilon=0.7)
# training_strategy_fn = lambda: EGreedyLinearStrategy(init_epsilon=1.0,
#                                                       min_epsilon=0.2,
#                                                       decay_steps=20000)
# training_strategy_fn = lambda: SoftMaxStrategy(init_temp=1.0,
#                                                min_temp=0.1,
#                                                exploration_ratio=0.8,
#                                                decay_steps=20000)
training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=1.0,
                                                  min_epsilon=0.3,
                                                  decay_steps=20000)
evaluation_strategy_fn = lambda: GreedyStrategy()

replay_buffer_fn = lambda: ReplayBuffer(max_size=10000, batch_size=64)
n_warmup_batches = 5
update_target_every_steps = 10

env_name, gamma, max_minutes, max_episodes, goal_mean_100_reward = environment_settings.values()
for seed in SEEDS:
    agent = DQN(replay_buffer_fn,
                value_model_fn,
                value_optimizer_fn,
                value_optimizer_lr,
                training_strategy_fn,
                evaluation_strategy_fn,
                n_warmup_batches,
                update_target_every_steps)

    make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)
    result, final_eval_score, training_time, wallclock_time = agent.train(
        make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    dqn_results.append(result)
    dqn_agents[seed] = agent
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_dqn_agent_key = seed
dqn_results = np.array(dqn_results)

#_ = BEEP()
#dqn_agents[best_dqn_agent_key].demo()


#nfq_root_dir = os.path.join(RESULTS_DIR, 'nfq')
#nfq_x = np.load(os.path.join(nfq_root_dir, 'x.npy'))
#
#nfq_max_r = np.load(os.path.join(nfq_root_dir, 'max_r.npy'))
#nfq_min_r = np.load(os.path.join(nfq_root_dir, 'min_r.npy'))
#nfq_mean_r = np.load(os.path.join(nfq_root_dir, 'mean_r.npy'))
#
#nfq_max_s = np.load(os.path.join(nfq_root_dir, 'max_s.npy'))
#nfq_min_s = np.load(os.path.join(nfq_root_dir, 'min_s.npy'))
#nfq_mean_s = np.load(os.path.join(nfq_root_dir, 'mean_s.npy'))
#
#nfq_max_t = np.load(os.path.join(nfq_root_dir, 'max_t.npy'))
#nfq_min_t = np.load(os.path.join(nfq_root_dir, 'min_t.npy'))
#nfq_mean_t = np.load(os.path.join(nfq_root_dir, 'mean_t.npy'))
#
#nfq_max_sec = np.load(os.path.join(nfq_root_dir, 'max_sec.npy'))
#nfq_min_sec = np.load(os.path.join(nfq_root_dir, 'min_sec.npy'))
#nfq_mean_sec = np.load(os.path.join(nfq_root_dir, 'mean_sec.npy'))
#
#nfq_max_rt = np.load(os.path.join(nfq_root_dir, 'max_rt.npy'))
#nfq_min_rt = np.load(os.path.join(nfq_root_dir, 'min_rt.npy'))
#nfq_mean_rt = np.load(os.path.join(nfq_root_dir, 'mean_rt.npy'))
#
#
#
#dqn_max_t, dqn_max_r, dqn_max_s,     dqn_max_sec, dqn_max_rt = np.max(dqn_results, axis=0).T
#dqn_min_t, dqn_min_r, dqn_min_s,     dqn_min_sec, dqn_min_rt = np.min(dqn_results, axis=0).T
#dqn_mean_t, dqn_mean_r, dqn_mean_s,     dqn_mean_sec, dqn_mean_rt = np.mean(dqn_results, axis=0).T
#dqn_x = np.arange(np.max((len(dqn_mean_s), len(nfq_mean_s))))
#
#plt.figure()
#
#fig, axs = plt.subplots(5, 1, figsize=(15,30), sharey=False, sharex=True)
#
## NFQ
#axs[0].plot(nfq_max_r, 'y', linewidth=1)
#axs[0].plot(nfq_min_r, 'y', linewidth=1)
#axs[0].plot(nfq_mean_r, 'y', label='NFQ', linewidth=2)
#axs[0].fill_between(nfq_x, nfq_min_r, nfq_max_r, facecolor='y', alpha=0.3)
#
#axs[1].plot(nfq_max_s, 'y', linewidth=1)
#axs[1].plot(nfq_min_s, 'y', linewidth=1)
#axs[1].plot(nfq_mean_s, 'y', label='NFQ', linewidth=2)
#axs[1].fill_between(nfq_x, nfq_min_s, nfq_max_s, facecolor='y', alpha=0.3)
#
#axs[2].plot(nfq_max_t, 'y', linewidth=1)
#axs[2].plot(nfq_min_t, 'y', linewidth=1)
#axs[2].plot(nfq_mean_t, 'y', label='NFQ', linewidth=2)
#axs[2].fill_between(nfq_x, nfq_min_t, nfq_max_t, facecolor='y', alpha=0.3)
#
#axs[3].plot(nfq_max_sec, 'y', linewidth=1)
#axs[3].plot(nfq_min_sec, 'y', linewidth=1)
#axs[3].plot(nfq_mean_sec, 'y', label='NFQ', linewidth=2)
#axs[3].fill_between(nfq_x, nfq_min_sec, nfq_max_sec, facecolor='y', alpha=0.3)
#
#axs[4].plot(nfq_max_rt, 'y', linewidth=1)
#axs[4].plot(nfq_min_rt, 'y', linewidth=1)
#axs[4].plot(nfq_mean_rt, 'y', label='NFQ', linewidth=2)
#axs[4].fill_between(nfq_x, nfq_min_rt, nfq_max_rt, facecolor='y', alpha=0.3)
#
## DQN
#axs[0].plot(dqn_max_r, 'b', linewidth=1)
#axs[0].plot(dqn_min_r, 'b', linewidth=1)
#axs[0].plot(dqn_mean_r, 'b--', label='DQN', linewidth=2)
#axs[0].fill_between(dqn_x, dqn_min_r, dqn_max_r, facecolor='b', alpha=0.3)
#
#axs[1].plot(dqn_max_s, 'b', linewidth=1)
#axs[1].plot(dqn_min_s, 'b', linewidth=1)
#axs[1].plot(dqn_mean_s, 'b--', label='DQN', linewidth=2)
#axs[1].fill_between(dqn_x, dqn_min_s, dqn_max_s, facecolor='b', alpha=0.3)
#
#axs[2].plot(dqn_max_t, 'b', linewidth=1)
#axs[2].plot(dqn_min_t, 'b', linewidth=1)
#axs[2].plot(dqn_mean_t, 'b--', label='DQN', linewidth=2)
#axs[2].fill_between(dqn_x, dqn_min_t, dqn_max_t, facecolor='b', alpha=0.3)
#
#axs[3].plot(dqn_max_sec, 'b', linewidth=1)
#axs[3].plot(dqn_min_sec, 'b', linewidth=1)
#axs[3].plot(dqn_mean_sec, 'b--', label='DQN', linewidth=2)
#axs[3].fill_between(dqn_x, dqn_min_sec, dqn_max_sec, facecolor='b', alpha=0.3)
#
#axs[4].plot(dqn_max_rt, 'b', linewidth=1)
#axs[4].plot(dqn_min_rt, 'b', linewidth=1)
#axs[4].plot(dqn_mean_rt, 'b--', label='DQN', linewidth=2)
#axs[4].fill_between(dqn_x, dqn_min_rt, dqn_max_rt, facecolor='b', alpha=0.3)
#
## ALL
#axs[0].set_title('Moving Avg Reward (Training)')
#axs[1].set_title('Moving Avg Reward (Evaluation)')
#axs[2].set_title('Total Steps')
#axs[3].set_title('Training Time')
#axs[4].set_title('Wall-clock Time')
#plt.xlabel('Episodes')
#axs[0].legend(loc='upper left')
#plt.show()




#dqn_root_dir = os.path.join(RESULTS_DIR, 'dqn')
#not os.path.exists(dqn_root_dir) and os.makedirs(dqn_root_dir)
#
#np.save(os.path.join(dqn_root_dir, 'x'), dqn_x)
#
#np.save(os.path.join(dqn_root_dir, 'max_r'), dqn_max_r)
#np.save(os.path.join(dqn_root_dir, 'min_r'), dqn_min_r)
#np.save(os.path.join(dqn_root_dir, 'mean_r'), dqn_mean_r)
#
#np.save(os.path.join(dqn_root_dir, 'max_s'), dqn_max_s)
#np.save(os.path.join(dqn_root_dir, 'min_s'), dqn_min_s )
#np.save(os.path.join(dqn_root_dir, 'mean_s'), dqn_mean_s)
#
#np.save(os.path.join(dqn_root_dir, 'max_t'), dqn_max_t)
#np.save(os.path.join(dqn_root_dir, 'min_t'), dqn_min_t)
#np.save(os.path.join(dqn_root_dir, 'mean_t'), dqn_mean_t)
#
#np.save(os.path.join(dqn_root_dir, 'max_sec'), dqn_max_sec)
#np.save(os.path.join(dqn_root_dir, 'min_sec'), dqn_min_sec)
#np.save(os.path.join(dqn_root_dir, 'mean_sec'), dqn_mean_sec)
#
#np.save(os.path.join(dqn_root_dir, 'max_rt'), dqn_max_rt)
#np.save(os.path.join(dqn_root_dir, 'min_rt'), dqn_min_rt)
#np.save(os.path.join(dqn_root_dir, 'mean_rt'), dqn_mean_rt)


# # Double DQN


pred = np.linspace(-100,100,500)
truth = np.zeros(pred.shape)
error = truth - pred




se = 0.5*error**2
ae = np.abs(error)
he = lambda delta=1: ae - delta/2 if delta == 0 else np.where(ae <= np.repeat(delta, len(ae)), se, delta*(ae - delta/2))



print(np.mean(se))
print(torch.Tensor(error).pow(2).mul(0.5).mean())




print(np.mean(ae))
print(torch.Tensor(error).abs().mean())




print(np.mean(he(float('inf'))))
print(np.mean(he(0)))



#plt.figure()
#plt.plot(pred, se)
#plt.title('Mean Squared Error (MSE/L2)')
#plt.show()
#
#
#plt.figure()
#plt.plot(pred, ae)
#plt.title('Mean Absolute Error (MAE/L1)')
#plt.show()
#
#
#
#plt.figure()
#plot1, = plt.plot(pred, he(30))
#plot2, = plt.plot(pred, he(10), ':')
#plt.title('Huber Loss')
#plt.legend([plot1,plot2],["Huber, δ=30", "Huber, δ=10"])
#plt.show()

plt.figure()
# plot1, = plt.plot(pred, se, ':')
plot1, = plt.plot(pred, he(float('inf')), ':')
plot2, = plt.plot(pred, he(30), '--')
plot3, = plt.plot(pred, he(10), '-.')
plot4, = plt.plot(pred, he(0))
# plot4, = plt.plot(pred, ae)
plt.title('MAE, MSE and Huber Loss')
plt.legend([plot1,plot2,plot3,plot4],["MSE/L2/Huber, δ=∞", "Huber, δ=30", "Huber, δ=10", "MAE/L1/Huber, δ=0"])
plt.show()




class DDQN():
    def __init__(self, 
                 reply_buffer_fn, 
                 value_model_fn, 
                 value_optimizer_fn, 
                 value_optimizer_lr,
                 max_gradient_norm,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps):
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.max_gradient_norm = max_gradient_norm
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)
        
        # argmax_a_q_sp = self.target_model(next_states).max(1)[1]
        argmax_a_q_sp = self.online_model(next_states).max(1)[1]
        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp[np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = self.online_model(states).gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()        
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 
                                       self.max_gradient_norm)
        self.value_optimizer.step()

    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, _ = env.step(action)
        past_limit_enforced = hasattr(env, '_past_limit') and env._past_limit()
        is_failure = is_terminal and not past_limit_enforced

        experience = (state, action, reward, new_state, float(is_failure))
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)
        return new_state, is_terminal
    
    def update_network(self):
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target.data.copy_(online.data)

    def train(self, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)
    
        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        self.episode_exploration = []
        
        self.target_model = self.value_model_fn(nS, nA)
        self.online_model = self.value_model_fn(nS, nA)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = training_strategy_fn()
        self.evaluation_strategy = evaluation_strategy_fn() 
                    
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()
            
            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                state, is_terminal = self.interaction_step(state, env)
                
                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)
                
                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_network()

                if is_terminal:
                    break
            
            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_model, env)
            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)
            
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:])/np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)
            
            wallclock_elapsed = time.time() - training_start
            result[episode-1] = total_step, mean_100_reward, mean_100_eval_score, training_time, wallclock_elapsed
            
#            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or                                reached_max_episodes or                                reached_goal_mean_reward
            elapsed_str = time.strftime("%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode-1, total_step, mean_10_reward, std_10_reward, 
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            if (episode-1)%25==0 or training_is_over:
                print(debug_message, end='\r', flush=True)
#            if reached_debug_time or training_is_over:
#                print(ERASE_LINE + debug_message, flush=True)
#                last_debug_time = time.time()
                
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break
                
        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
        env.close() ; del env
        return result, final_eval_score, training_time, wallclock_time
    
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(self.online_model, s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)
    
    def demo(self, title='Trained {} Agent', n_episodes=10, max_n_videos=3):          
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation')
        self.evaluate(self.online_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)




ddqn_results = []
ddqn_agents, best_ddqn_agent_key, best_eval_score = {}, None, float('-inf')
for seed in SEEDS:
    environment_settings = {
        'env_name': 'CartPole-v1',
        'gamma': 1.00,
        'max_minutes': 20,
        'max_episodes': 10000,
        'goal_mean_100_reward': 400
    }

    value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512,128))
    value_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005
    max_gradient_norm = float('inf')

    training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=1.0,  
                                                      min_epsilon=0.3, 
                                                      decay_steps=20000)
    evaluation_strategy_fn = lambda: GreedyStrategy()

    replay_buffer_fn = lambda: ReplayBuffer(max_size=50000, batch_size=64)
    n_warmup_batches = 5
    update_target_every_steps = 10
    
    env_name, gamma, max_minutes, max_episodes, goal_mean_100_reward = environment_settings.values()
    agent = DDQN(replay_buffer_fn, 
                 value_model_fn, 
                 value_optimizer_fn, 
                 value_optimizer_lr,
                 max_gradient_norm,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps)

    make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)
    result, final_eval_score, training_time, wallclock_time = agent.train(
        make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    ddqn_results.append(result)
    ddqn_agents[seed] = agent
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_ddqn_agent_key = seed
ddqn_results = np.array(ddqn_results)
#_ = BEEP()




#ddqn_agents[best_ddqn_agent_key].demo()




ddqn_max_t, ddqn_max_r, ddqn_max_s, ddqn_max_sec, ddqn_max_rt = np.max(ddqn_results, axis=0).T
ddqn_min_t, ddqn_min_r, ddqn_min_s, ddqn_min_sec, ddqn_min_rt = np.min(ddqn_results, axis=0).T
ddqn_mean_t, ddqn_mean_r, ddqn_mean_s, ddqn_mean_sec, ddqn_mean_rt = np.mean(ddqn_results, axis=0).T
ddqn_x = np.arange(np.max((len(ddqn_mean_s), len(dqn_mean_s))))




#fig, axs = plt.subplots(5, 1, figsize=(15,30), sharey=False, sharex=True)
#
## DQN
#axs[0].plot(dqn_max_r, 'b', linewidth=1)
#axs[0].plot(dqn_min_r, 'b', linewidth=1)
#axs[0].plot(dqn_mean_r, 'b--', label='DQN', linewidth=2)
#axs[0].fill_between(dqn_x, dqn_min_r, dqn_max_r, facecolor='b', alpha=0.3)
#
#axs[1].plot(dqn_max_s, 'b', linewidth=1)
#axs[1].plot(dqn_min_s, 'b', linewidth=1)
#axs[1].plot(dqn_mean_s, 'b--', label='DQN', linewidth=2)
#axs[1].fill_between(dqn_x, dqn_min_s, dqn_max_s, facecolor='b', alpha=0.3)
#
#axs[2].plot(dqn_max_t, 'b', linewidth=1)
#axs[2].plot(dqn_min_t, 'b', linewidth=1)
#axs[2].plot(dqn_mean_t, 'b--', label='DQN', linewidth=2)
#axs[2].fill_between(dqn_x, dqn_min_t, dqn_max_t, facecolor='b', alpha=0.3)
#
#axs[3].plot(dqn_max_sec, 'b', linewidth=1)
#axs[3].plot(dqn_min_sec, 'b', linewidth=1)
#axs[3].plot(dqn_mean_sec, 'b--', label='DQN', linewidth=2)
#axs[3].fill_between(dqn_x, dqn_min_sec, dqn_max_sec, facecolor='b', alpha=0.3)
#
#axs[4].plot(dqn_max_rt, 'b', linewidth=1)
#axs[4].plot(dqn_min_rt, 'b', linewidth=1)
#axs[4].plot(dqn_mean_rt, 'b--', label='DQN', linewidth=2)
#axs[4].fill_between(dqn_x, dqn_min_rt, dqn_max_rt, facecolor='b', alpha=0.3)
#
## DDQN
#axs[0].plot(ddqn_max_r, 'g', linewidth=1)
#axs[0].plot(ddqn_min_r, 'g', linewidth=1)
#axs[0].plot(ddqn_mean_r, 'g-.', label='DDQN', linewidth=2)
#axs[0].fill_between(ddqn_x, ddqn_min_r, ddqn_max_r, facecolor='g', alpha=0.3)
#
#axs[1].plot(ddqn_max_s, 'g', linewidth=1)
#axs[1].plot(ddqn_min_s, 'g', linewidth=1)
#axs[1].plot(ddqn_mean_s, 'g-.', label='DDQN', linewidth=2)
#axs[1].fill_between(ddqn_x, ddqn_min_s, ddqn_max_s, facecolor='g', alpha=0.3)
#
#axs[2].plot(ddqn_max_t, 'g', linewidth=1)
#axs[2].plot(ddqn_min_t, 'g', linewidth=1)
#axs[2].plot(ddqn_mean_t, 'g-.', label='DDQN', linewidth=2)
#axs[2].fill_between(ddqn_x, ddqn_min_t, ddqn_max_t, facecolor='g', alpha=0.3)
#
#axs[3].plot(ddqn_max_sec, 'g', linewidth=1)
#axs[3].plot(ddqn_min_sec, 'g', linewidth=1)
#axs[3].plot(ddqn_mean_sec, 'g-.', label='DDQN', linewidth=2)
#axs[3].fill_between(ddqn_x, ddqn_min_sec, ddqn_max_sec, facecolor='g', alpha=0.3)
#
#axs[4].plot(ddqn_max_rt, 'g', linewidth=1)
#axs[4].plot(ddqn_min_rt, 'g', linewidth=1)
#axs[4].plot(ddqn_mean_rt, 'g-.', label='DDQN', linewidth=2)
#axs[4].fill_between(ddqn_x, ddqn_min_rt, ddqn_max_rt, facecolor='g', alpha=0.3)
#
## ALL
#axs[0].set_title('Moving Avg Reward (Training)')
#axs[1].set_title('Moving Avg Reward (Evaluation)')
#axs[2].set_title('Total Steps')
#axs[3].set_title('Training Time')
#axs[4].set_title('Wall-clock Time')
#plt.xlabel('Episodes')
#axs[0].legend(loc='upper left')
#plt.show()



#
#ddqn_root_dir = os.path.join(RESULTS_DIR, 'ddqn')
#not os.path.exists(ddqn_root_dir) and os.makedirs(ddqn_root_dir)
#
#np.save(os.path.join(ddqn_root_dir, 'x'), ddqn_x)
#
#np.save(os.path.join(ddqn_root_dir, 'max_r'), ddqn_max_r)
#np.save(os.path.join(ddqn_root_dir, 'min_r'), ddqn_min_r)
#np.save(os.path.join(ddqn_root_dir, 'mean_r'), ddqn_mean_r)
#
#np.save(os.path.join(ddqn_root_dir, 'max_s'), ddqn_max_s)
#np.save(os.path.join(ddqn_root_dir, 'min_s'), ddqn_min_s )
#np.save(os.path.join(ddqn_root_dir, 'mean_s'), ddqn_mean_s)
#
#np.save(os.path.join(ddqn_root_dir, 'max_t'), ddqn_max_t)
#np.save(os.path.join(ddqn_root_dir, 'min_t'), ddqn_min_t)
#np.save(os.path.join(ddqn_root_dir, 'mean_t'), ddqn_mean_t)
#
#np.save(os.path.join(ddqn_root_dir, 'max_sec'), ddqn_max_sec)
#np.save(os.path.join(ddqn_root_dir, 'min_sec'), ddqn_min_sec)
#np.save(os.path.join(ddqn_root_dir, 'mean_sec'), ddqn_mean_sec)
#
#np.save(os.path.join(ddqn_root_dir, 'max_rt'), ddqn_max_rt)
#np.save(os.path.join(ddqn_root_dir, 'min_rt'), ddqn_min_rt)
#np.save(os.path.join(ddqn_root_dir, 'mean_rt'), ddqn_mean_rt)




env = make_env_fn(**make_env_kargs, seed=123, monitor_mode='evaluation')
state = env.reset()
env.close()
del env
print(state)




q_values = ddqn_agents[best_ddqn_agent_key].online_model(state).detach().cpu().numpy()[0]
print(q_values)




q_s = q_values
v_s = q_values.mean()
a_s = q_values - q_values.mean()



plt.figure()
plt.bar(('Left (idx=0)','Right (idx=1)'), q_s)
plt.xlabel('Action')
plt.ylabel('Estimate')
plt.title("Action-value function, Q(" + str(np.round(state,2)) + ")")
plt.show()



plt.figure()
plt.bar('s='+str(np.round(state,2)), v_s, width=0.1)
plt.xlabel('State')
plt.ylabel('Estimate')
plt.title("State-value function, V("+str(np.round(state,2))+")")
plt.show()



plt.figure()
plt.bar(('Left (idx=0)','Right (idx=1)'), a_s)
plt.xlabel('Action')
plt.ylabel('Estimate')
plt.title("Advantage function, (" + str(np.round(state,2)) + ")")
plt.show()




env = make_env_fn(**make_env_kargs, seed=123, monitor_mode='evaluation')

states = []
for agent in ddqn_agents.values():
    for episode in range(100):
        state, done = env.reset(), False
        while not done:
            states.append(state)
            action = agent.evaluation_strategy.select_action(agent.online_model, state)
            state, _, done, _ = env.step(action)
env.close()
del env

x = np.array(states)[:,0]
xd = np.array(states)[:,1]
a = np.array(states)[:,2]
ad = np.array(states)[:,3]



plt.figure()
parts = plt.violinplot((x, xd, a, ad), 
                       vert=False, showmeans=False, showmedians=False, showextrema=False)

colors = ['red','green','yellow','blue']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor(colors[i])
    pc.set_alpha(0.5)

plt.yticks(range(1,5), ["cart position", "cart velocity", "pole angle", "pole velocity"])
plt.yticks(rotation=45)
plt.title('Range of state-variable values for ' + str(
    ddqn_agents[best_ddqn_agent_key].__class__.__name__))

plt.show()


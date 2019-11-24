#!/usr/bin/env python
# coding: utf-8



import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import matplotlib.pyplot as plt
from skimage.transform import resize
from collections import deque
from IPython.display import clear_output, display
import torch.multiprocessing as mp

import time

#env = gym.make("Pong-v0")
env = gym.make("CartPole-v1")
env.reset()
#env.unwrapped.get_action_meanings()
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4,25)
        self.l2 = nn.Linear(25,50)
        self.actor_lin1 = nn.Linear(50,2)
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25,1)

    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y),dim=0)
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic




def evaluate(worker_model):
    test_env = gym.make("CartPole-v1")
    test_env.reset()
    maxrun = 0
    done = False
    env.reset()
    raw_state = np.array(test_env.env.state)
    state = torch.from_numpy(raw_state).float()
    while(done==False):
        #env.render('human')
        policy, value = worker_model(state)
        #sample action
        action = torch.distributions.Categorical(logits=policy.view(-1)).sample().detach().numpy()
        state_, reward, done, lives = test_env.step(action)
        #print(value,reward)
        state = torch.from_numpy(state_).float()
        maxrun += 1
    test_env.close()
    return maxrun

def update_params(worker_opt,values,logprobs,rewards,G,clc=0.1,gamma=0.95):
        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1) #to Tensor and reverse
        values = torch.stack(values).flip(dims=(0,)).view(-1) #to Tensor and reverse
        Returns = []
        ret_ = G
        for r in range(rewards.shape[0]):
            ret_ = rewards[r] + gamma * ret_
            Returns.append(ret_)
        Returns = torch.stack(Returns).view(-1)
        Returns = F.normalize(Returns,dim=0)
        actor_loss = -1*logprobs * (Returns - values.detach())
        critic_loss = torch.pow(values - Returns,2)
        loss = actor_loss.sum() + clc*critic_loss.sum()
        worker_opt.zero_grad()
        loss.backward()
        worker_opt.step()
        return actor_loss, critic_loss
        
def run_episode(worker_env, worker_model, N_steps=10):
    raw_state = np.array(worker_env.env.state)
    state = torch.from_numpy(raw_state).float()
    values, logprobs, rewards = [],[],[]
    done = False
    j=0
    G=torch.Tensor([0])
    while (j < N_steps and done == False):
        j+=1
        #run actor critic model
        policy, value = worker_model(state)
        values.append(value)
        #sample action
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, info = worker_env.step(action.detach().numpy())
        #reward = reward * 10
        state = torch.from_numpy(state_).float()
        if done:
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
            #_,value = worker_model(state)
            G = value.detach()
        rewards.append(reward)
    return values, logprobs, rewards, G

def worker(t, worker_model, counter, params, eplens): #q is mp Queue
    start_time = time.time()
    print("In process {}".format(t,))
    #play n steps of the game, store rewards
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters())
    # worker_opt.zero_grad()
    maxrun = 1
    for i in range(params['epochs']):
        # worker_opt.zero_grad()
        #stores
        values, logprobs, rewards, G = run_episode(worker_env,worker_model, params['n_steps'])
        actor_loss, critic_loss = update_params(worker_opt,values,logprobs,rewards,G)
        counter.value = counter.value + 1
        if i % 50 == 0:
            eplen = evaluate(worker_model)
            eplens.put(eplen)
            print("Process: {} Epoch: {} Maxrun: {} ALoss: {} CLoss: {}".format(t, i, eplen, actor_loss.detach().mean().numpy(), critic_loss.detach().mean().numpy()))
        if time.time() - start_time > 300:
            print("Done 60 seconds")
            break;




'''%%time
TestModel = ActorCritic()
worker_opt = optim.Adam(lr=1e-4,params=TestModel.parameters())
q2 = mp.Value('i',0)
params = {
    'epochs':5,
    'n_steps':5,
    'n_workers':1,
}
AC_step(0,TestModel,q2,params)'''


if __name__ == '__main__':
    MasterNode = ActorCritic()
    MasterNode.share_memory()
    processes = []
    #worker_opt = optim.Adam(lr=1e-4,params=MasterNode.parameters())
    params = {
        'epochs':1500000,
        'n_steps':10,
        'n_workers':1,
    }
    counter = mp.Value('i',0)
    eplens = mp.Queue()
    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i,MasterNode,counter,params,eplens))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()
        
    print(counter.value,processes[0].exitcode)
    
    
    
    eplens_ = []
    while not eplens.empty():
        eplens_.append(eplens.get())
    
    
    
    
    plt.figure(figsize=(9,5))
    x = np.array(eplens_)
    N = 50
    x = np.convolve(x, np.ones((N,))/N, mode='valid')
    plt.ylabel("Mean Episode Length")
    plt.xlabel("Training Time")
    plt.title("CartPole Training Evaluation")
    plt.plot(x)
    #plt.savefig("avg_rewards_Nstep.pdf")
    
    
    # ## Test
    
    
    
    steps = 2000
    env = gym.make("CartPole-v1")
    env.reset()
    maxrun = 0
    state = torch.from_numpy(env.env.state).float()
    done = False
    avg_run = 0
    runs = int(100)
    for i in range(runs):
        maxrun = 0
        done = False
        env.reset()
        state = torch.from_numpy(env.env.state).float()
        while(done==False):
            #env.render('human')
            policy, value = MasterNode(state)
            #sample action
            action = torch.distributions.Categorical(logits=policy.view(-1)).sample().detach().numpy()
            state_, reward, done, lives = env.step(action)
            
            #print(value,reward)
            state = torch.from_numpy(state_).float()
            maxrun += 1
        avg_run += maxrun
    avg_run = avg_run / runs
    env.close()
    print("Maxrun: {}".format(avg_run,))
    
    
    
    
    '''TestModel = ActorCritic()
    env = gym.make("CartPole-v1")
    env.reset()
    maxrun = 0
    state = torch.from_numpy(env.env.state).float()
    done = False
    avg_run = 0
    runs = int(200)
    for i in range(runs):
        maxrun = 0
        done = False
        env.reset()
        state = torch.from_numpy(env.env.state).float()
        while(done==False):
            #env.render('human')
            policy, value = TestModel(state)
            #sample action
            action = torch.distributions.Categorical(logits=policy.view(-1)).sample()
            state_, reward, done, lives = env.step(env.action_space.sample())
            state = torch.from_numpy(state_).float()
            maxrun += 1
        avg_run += maxrun
    avg_run /= runs
    env.close()
    print("Maxrun: {}".format(avg_run,))'''
    
    
    # ### Demonstrating how bootstrapping reduces variance
    
    
    
    r1 = [1,1,-1]
    r2 = [1,1,1]
    R1,R2 = 0.0,0.0
    #No bootstrapping
    for i in range(len(r1)-1,0,-1):
        R1 = r1[i] + 0.99*R1
    for i in range(len(r2)-1,0,-1):
        R2 = r2[i] + 0.99*R2
    print("No bootstrapping")
    print(R1,R2)
    #With bootstrapping
    R1,R2 = 1.0,1.0
    for i in range(len(r1)-1,0,-1):
        R1 = r1[i] + 0.99*R1
    for i in range(len(r2)-1,0,-1):
        R2 = r2[i] + 0.99*R2
    print("With bootstrapping")
    print(R1,R2)








#PS1 IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import sklearn
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 

from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn import preprocessing

#KNN imports
from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

#SVM
from sklearn import svm

from sklearn.model_selection import GridSearchCV


#PS2 IMPORTS
import mlrose
#import numpy as np
import random
import mlrose_hiive as mlrh

#PS3 Imports
import sklearn.cluster
import sklearn.mixture
from matplotlib.colors import LogNorm
import sklearn.decomposition
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import random_projection
from sklearn.random_projection import johnson_lindenstrauss_min_dim


#PS4 Imports
import numpy as np
import gym
import random
import time
from IPython.display import clear_output

#import mdptoolbox, mdptoolbox.example
import hiive.mdptoolbox
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp
from gym import wrappers
from gym.envs.toy_text.frozen_lake import generate_random_map

import seaborn as sns

def main():
    return


def run_episode(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
#    print('num steps',step_idx)
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
#    print('scores',scores)
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    print('optimal policy',policy)
    return policy


def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
    
    v = np.zeros(env.nS)  # initialize value-function
    max_iterations = 10000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.nS):
#original implementation. might be wrong bc it doesn't have a gamma discounting the future rewards            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
            q_sa = [sum([p*(r + gamma*prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            print('v',v)
            break
    return v

def custom_map():
    #Do the custom map as the "large MDP", then use random maps to do analysis on the sizing
    '''
    custom_frozen_lake = [
    'SFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFFFFFFG'
    ]    
    '''
    custom_frozen_lake = [
    'SFFHFFFFFFFFFFHFFFFF',
    'FFFHFFFFFFHFFFFFFFFF',
    'HFFFFFFFHFFFFFFFFFFF',
    'FFFFFHFFFFFFFFFFHFFF',
    'FFFFFFFFHFFFHFFFFFFF',
    'HFFFFFFFFFFFFFFFFFFH',
    'FFFFHFFFFFFFFHFFFFFF',
    'FFFFHFFFHFFFFFFFFFFF',
    'FFFHFFFFFFFFFFFFFHFF',
    'FHFFFFHFFFFFFFFFFFFF',
    'FFFFFFFFFFHFFFFFFFHF',
    'FFFFFFFFFFFFFHFFFFHF',
    'FFFFHFFFFHFFFFFFFFFF',
    'FFHFFFFFFFFHFFFFFFFF',
    'FFFFFFHFFFFFFFFFFFHF',
    'FFFFFFFFFFFHFFFFFFHF',
    'FFFHFFHFFFFFFFFFFFFF',
    'HFFFHFFFFFFFFFFFFFFF',
    'HFFFFFFFFHFFFFFFFFFF',
    'FFFFHFFHFFFFFFFFFFFG'
    ]    
    return custom_frozen_lake

if __name__ == '__main__':
#    random_map=generate_random_map(size=4,p=0.9)
#    custom_fl=custom_map()
#    env=gym.make("FrozenLake-v0",desc=custom_fl)#desc=random_map)
    env=gym.make("FrozenLake8x8-v0")
#    env_name  = 'FrozenLake-v0'
    env._max_episode_steps=1000  #Required or the environment won't process for more than 100 steps and the values will be all fucked upppppppp
#    print('env.P',env.P)
#    raise NotImplementedError
    env.render()
    gamma = 0.99
#    env = gym.make(env_name)

    
    #Run Environment
    a_time=time.time()
    optimal_v = value_iteration(env, gamma);
    b_time=time.time()
    policy = extract_policy(optimal_v, gamma)
    policy_score = evaluate_policy(env, policy, gamma, n=1000)

    #Optimal Policy Heatmap
    ax=sns.heatmap(policy.reshape(int(np.sqrt(env.nS)),int(np.sqrt(env.nS))),cmap="YlGnBu", annot=True, cbar=False)
    ax.set_title('Value Iteration -- Optimal Policy')
    plt.show()
    '''
    #Environment Heatmap
    map_as_array=np.array(custom_fl)
    print('custom_fl',custom_fl)
    print('shape',map_as_array.shape)
    ax=sns.heatmap(map_as_array.reshape(20,20),cmap="YlGnBu", annot=True, cbar=False)
    ax.set_title('Frozen Lake - Environment')
    plt.show()
    '''
    #Value Heatmap
    sns.set(font_scale=0.5)
    ax=sns.heatmap(optimal_v.reshape(int(np.sqrt(env.nS)),int(np.sqrt(env.nS))),cmap="YlGnBu", annot=True, cbar=False)
    ax.set_title('Value Iteration -- State Values')
    plt.show()
    
    print('Policy average score = ', policy_score)
    print('Time to Convergence: ', b_time-a_time)


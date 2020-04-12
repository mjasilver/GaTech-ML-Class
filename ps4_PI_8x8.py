
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

import seaborn as sns
"""
Solving FrozenLake8x8 environment using Policy iteration.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
#Code can be found here: https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
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
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0,print_map=False):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    """
    v = np.zeros(env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break

    if print_map==True:
    #Value Heatmap
        sns.set(font_scale=0.5)
        ax=sns.heatmap(v.reshape(int(np.sqrt(env.nS)),int(np.sqrt(env.nS))),cmap="YlGnBu", annot=True, cbar=False)
        ax.set_title('Policy Iteration -- State Values')
        plt.show()

    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    max_iterations = 200000
#    gamma = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy

    heat_map_policy_display = compute_policy_v(env,policy, gamma,print_map=True)
    
    return policy

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
    env_name  = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
#    custom_fl=custom_map()
#    env=gym.make("FrozenLake-v0",desc=custom_fl)#desc=random_map)
    env._max_episode_steps=1000  #Required or the environment won't process for more than 100 steps and the values will be all fucked upppppppp

    gamma=0.99
    a_time=time.time()
    optimal_policy = policy_iteration(env, gamma = gamma) #gamma=1.0)
    b_time=time.time()
    scores = evaluate_policy(env, optimal_policy, gamma = gamma) #gamma=1.0)
#    print('Average scores = ', np.mean(scores))


    #Optimal Policy Heatmap
    ax=sns.heatmap(optimal_policy.reshape(int(np.sqrt(env.nS)),int(np.sqrt(env.nS))),cmap="YlGnBu", annot=True, cbar=False)
    ax.set_title('Policy Iteration -- Optimal Policy')
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
#Moved this to within the policy_iteration() method
#    sns.set(font_scale=0.5)
#    ax=sns.heatmap(optimal_v.reshape(20,20),cmap="YlGnBu", annot=True, cbar=False)
#    ax.set_title('Value Iteration -- State Values')
#    plt.show()
    
    print('Average scores = ', np.mean(scores))
    print('Time to Convergence: ', b_time-a_time)


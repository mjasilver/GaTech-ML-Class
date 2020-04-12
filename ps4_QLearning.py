
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
    #Exponential Decay
#    q_learning(game="FrozenLake-v0",num_episodes=100000,max_timesteps=1000,learning_rate=0.1,discount_rate=0.99,exploration_rate=1,exploration_decay_rate=0.001,
#                    start_exploration_rate=1.0,end_exploration_rate=0.01, decay='exponential')
    #Linear Decay
    q_learning(game="FrozenLake-v0",num_episodes=10000,max_timesteps=1000,learning_rate=0.1,discount_rate=0.99,exploration_rate=1,exploration_decay_rate=0.001,
                    start_exploration_rate=1.0,end_exploration_rate=0.01, decay='linear',enhanced_exploitation=True)
    return

    

def q_learning(game="FrozenLake-v0",num_episodes=100000,max_timesteps=100,learning_rate=0.1,discount_rate=0.99,exploration_rate=1,exploration_decay_rate=0.001,
                    start_exploration_rate=1.0,end_exploration_rate=0.01,decay='exponential',enhanced_exploitation=False):
    print('******NEED TO REWRITE THIS COOODDEEEE**********')


    custom_fl=custom_map()
    env=gym.make("FrozenLake-v0",desc=custom_fl)#desc=random_map)
    env._max_episode_steps=1000  #Required or the environment won't process for more than 100 steps and the values will be all fucked upppppppp
##    random_map=generate_random_map(size=20,p=0.98)
##    env=gym.make("FrozenLake-v0",desc=random_map)    
###    env=gym.make("FrozenLake8x8-v0")    


    env._max_episode_steps=5000  #Required or the environment won't process for more than 100 steps and the values will be all fucked upppppppp

    #Create Q-Table
    action_space=env.action_space.n
    print('action space size',env.action_space)
    state_space=env.observation_space.n
    print('observation space size',env.observation_space)
    q_table = np.zeros((state_space,action_space))
    state_visitation_table=np.zeros((state_space,1))
    state_action_exploration_table= np.zeros((state_space,action_space))
    termination_state_actions=np.zeros((state_space,action_space))

    #Create reward tracking array
    reward_by_episode=[]
    exploration_by_episode=[]


    a_time=time.time()
    for i_episode in range(num_episodes):
        state = env.reset()
        reward_total = 0
        done = False

        for t in range(max_timesteps):
            state_visitation_table[state]=state_visitation_table[state]+1
            #Choose the argmax action or a random action
            random_num = random.uniform(0,1)
            if random_num < exploration_rate:# and (enhanced_exploitation==False or state_visitation_table[state]<1000): #If enhanced_exploitation is True and you've seen that state >1000 times, you don't want to explore. So, when these are both False, exploit
                action=env.action_space.sample()        #Random action
            #elif enhanced_exploitation==True and np.min(state_action_exploration_table[state][:])<10000:   #if you've seen a state-action less than 1000 times, take the min action 
            #    action=np.argmin(state_action_exploration_table[state][:])  #Pick the action you've done the least of
            #elif np.max(q_table[state,:])==0:
            #    action=env.action_space.sample()
            else:
                action = np.argmax(q_table[state,:])    #Argmax action

            if termination_state_actions[state][action]/np.sum(termination_state_actions[state][:])>.50 and q_table[state][action]<0.5:  #If this is a hole, rerandomize!!!
                action=env.action_space.sample()

            state_action_exploration_table[state][action]=state_action_exploration_table[state][action]+1

            #Take the next step
            new_state,reward,done,info = env.step(action)

            #Update the Q-Table
            q_table[state,action]=(1-learning_rate)*q_table[state,action] + learning_rate*(reward+discount_rate*np.max(q_table[new_state,:]))

            state=new_state
            reward_total = reward_total + reward

            if done:
#                print("Episode finished after {} timesteps".format(t+1))
                termination_state_actions[state][action] += termination_state_actions[state][action]
                break

        #Decay the exploration rate
        if decay=='exponential':
            exploration_rate=end_exploration_rate + (start_exploration_rate - end_exploration_rate)*np.exp(-exploration_decay_rate*i_episode)
            reward_by_episode.append(reward_total)
            exploration_by_episode.append(exploration_rate)

        #Decay the exploration rate
        if decay=='linear':
            exploration_rate=start_exploration_rate-i_episode*(start_exploration_rate-end_exploration_rate)/num_episodes
            reward_by_episode.append(reward_total)
            exploration_by_episode.append(exploration_rate)

    env.close()
    b_time=time.time()

    #Print updated qtable
    print("----Q-TABLE----")
    print(q_table)

    #Calc and print reward/1000 episodes
    rewards_per_thousand_episodes=np.split(np.array(reward_by_episode),num_episodes/1000)
    exploration_per_thousand_episodes=np.split(np.array(exploration_by_episode),num_episodes/1000)

    print("AVE EXPLORATION RATE BY EPISODE(K)")
    count=1000
    for expl in exploration_per_thousand_episodes:
        print(count,": ",str(sum(expl/1000)))
        count += 1000

    count=1000
    print("REWARD AVE BY EPISODE(K)")
    for r in rewards_per_thousand_episodes:
        print(count,": ",str(sum(r/1000)))
        count += 1000


    print('Q_LEARNING TIME TO CONVERGENCE',b_time-a_time)

    policy,value_function=extract_policy_and_value_functions(env,q_table)

    
    #Optimal Policy Heatmap
    ax=sns.heatmap(policy.reshape(int(np.sqrt(env.nS)),int(np.sqrt(env.nS))),cmap="YlGnBu", annot=True, cbar=False)
    ax.set_title('Q-Learning -- Optimal Policy')
    plt.show()

    #Value Heatmap
    sns.set(font_scale=0.5)
    ax=sns.heatmap(value_function.reshape(int(np.sqrt(env.nS)),int(np.sqrt(env.nS))),cmap="YlGnBu", annot=True, cbar=False)
    ax.set_title('Q-Learning -- State Values')
    plt.show()
    
    #State Visitation Heatmap
    sns.set(font_scale=0.33)
    ax=sns.heatmap(state_visitation_table.reshape(int(np.sqrt(env.nS)),int(np.sqrt(env.nS))),cmap="YlGnBu", annot=True, cbar=False)
    ax.set_title('Q-Learning -- Number of Times State is Sampled')
    plt.show()
    
    #Chart 1: Value Function, S=3 for different discount factors
#    discount_array=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    plt.plot(range(num_episodes),exploration_by_episode)#,discount_array,entropy_base)#(range(1,number_components),silhouette_vector)
    plt.title('Q-Learning: Exploration Rate by Episode')#'Information Gain by Cluster'
    plt.xlabel('Episode Number')
    plt.ylabel('Exploration Rate')
    plt.legend([decay])
#    plt.ylim((0,1.0))
    plt.show()

    rewards_policy=run_frozenlake_policy(env,1000,max_timesteps,q_table,discount_rate)
    print('AVE REWARD FOR LEARNED POLICY:',np.mean(rewards_policy))

    return

def extract_policy_and_value_functions(env,q_table):
    policy=np.zeros((env.nS,1))
    value_function=np.zeros((env.nS,1))
    for s in range(env.nS):
        policy[s]=np.argmax(q_table[s][:])
        value_function[s]=np.max(q_table[s][:])
#    print('extracted policy')
#    print('policy',policy)
#    print('value function',value_function)
#    print('Q-Table',q_table)
    return policy,value_function

def run_frozenlake_policy(env,num_episodes,max_timesteps,q_table,discount_rate):
    #Create reward tracking array
    reward_by_episode=[]

    print('max_timesteps',max_timesteps)
    a_time=time.time()
    for i_episode in range(num_episodes):
        state = env.reset()
        reward_total = 0
        done = False

        for t in range(max_timesteps):

            #Choose the argmax action
            action = np.argmax(q_table[state,:])    #Argmax action

            #Take the next step
            new_state,reward,done,info = env.step(action)

            state=new_state
            reward_total = reward_total + reward*discount_rate**t
            if done:
#                print("Episode finished after {} timesteps".format(t+1))
                reward_by_episode.append(reward_total)
                break

    env.close()
    return reward_by_episode


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



if __name__ == "__main__": 			  		 			 	 	 		 		 	  		   	  			  	
    main()

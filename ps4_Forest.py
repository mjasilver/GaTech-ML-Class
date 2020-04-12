
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

import mdptoolbox
import mdptoolbox.example
import mdptoolbox.mdp

def main():
#    vi_forest()
#    pi_forest()
    qlearning_forest()
    return

def vi_forest():
    print('---VALUE ITERATION FOREST---')
    P,R = hiive.mdptoolbox.example.forest(S=3,r1=4,r2=2,p=0.1,is_sparse=False)
    vi=hiive.mdptoolbox.mdp.ValueIteration(P,R,0.5,epsilon=0.5)
#    vi=hiive.mdptoolbox.mdp.ValueIteration(P,R,0.5,epsilon=0.7) #--> This epsilon value works with the given discount value, but not other discount values for some reason. Not really sure why
#    vi=hiive.mdptoolbox.mdp.ValueIterationGS(P,R,0.96,epsilon=0.01)
    vi.setVerbose()
    vi.run()

    print('P',vi.P)
    print('R',vi.R)

    print('VI Policy',vi.policy)
    print('VI V',vi.V)
    print('VI Iter',vi.iter)
    print('VI Time',vi.time)


    discount_array=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    value_array=[]
    P,R = hiive.mdptoolbox.example.forest(S=3,r1=4,r2=2,p=0.1,is_sparse=False)
    print('')
    print('Optimal Policies')
    for d in discount_array:
        vi=hiive.mdptoolbox.mdp.ValueIteration(P,R,d,epsilon=0.5)
        vi.run()
#        print(type(vi.V),vi.V)
        value_array.append(vi.V[2])        
        print('VI discount: ',d, 'Policy: ',vi.policy)

    #Chart 1: Value Function, S=3 for different discount factors
#    discount_array=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    plt.plot(discount_array,value_array)#,discount_array,entropy_base)#(range(1,number_components),silhouette_vector)
    plt.title('Value Iteration: Value of State 3, by Discount Factor (Gamma)')#'Information Gain by Cluster'
    plt.xlabel('Discount Factor - Gamma')
    plt.ylabel('Value')
#    plt.legend(['Clusters','Baseline'])
#    plt.ylim((0,1.0))
    plt.show()

    return

def pi_forest():
    print('---POLICY ITERATION FOREST---')
    P,R = hiive.mdptoolbox.example.forest()
    pi=hiive.mdptoolbox.mdp.PolicyIteration(P,R,0.5)#,epsilon=8)
    pi.setVerbose()
    pi.run()

    print('P',pi.P)
    print('R',pi.R)
    
    print('PI Policy',pi.policy)
    print('PI V',pi.V)
    print('PI Iter',pi.iter)
    print('PI Time',pi.time)

    
    discount_array=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    value_array=[]
    P,R = hiive.mdptoolbox.example.forest(S=3,r1=4,r2=2,p=0.1,is_sparse=False)
    print('')
    print('Optimal Policies')
    for d in discount_array:
        pi=hiive.mdptoolbox.mdp.PolicyIteration(P,R,d)
        pi.run()
#        print(type(vi.V),vi.V)
        value_array.append(pi.V[2])
        print('PI discount: ',d, '     Policy: ',pi.policy)
    print('')
    
    #Chart 1: Value Function, S=3 for different discount factors
#    discount_array=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    plt.plot(discount_array,value_array)#,discount_array,entropy_base)#(range(1,number_components),silhouette_vector)
    plt.title('Policy Iteration: Value of State 3, by Discount Factor (Gamma)')#'Information Gain by Cluster'
    plt.xlabel('Discount Factor - Gamma')
    plt.ylabel('Value')
#    plt.legend(['Clusters','Baseline'])
#    plt.ylim((0,1.0))
    plt.show()

    return

def qlearning_forest():
    print('---Q_LEARNING FOREST---')
#    np.random.seed(0)
    P,R = hiive.mdptoolbox.example.forest()
    print('Transition Function')
    print(P)
    print('Reward Function')
    print(R)
    ql=hiive.mdptoolbox.mdp.QLearning(P,R,0.5,n_iter=100000)
    a=time.time()
    ql.run()
    b=time.time()
    print('QL Policy',ql.policy)
    print('QL V',ql.V)
    print('QL Q',ql.Q)
    print('QL Time',b-a)
#    print('QL Iter',vi.iter)
#    print('QL Time',vi.time)
#    print('Mean Discrepancy',ql.mean_discrepancy)


    discount_array=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    value_array=[]
    P,R = hiive.mdptoolbox.example.forest(S=3,r1=4,r2=2,p=0.1,is_sparse=False)
    print('')
    print('Optimal Policies')
    for d in discount_array:
        ql=hiive.mdptoolbox.mdp.QLearning(P,R,d,n_iter=100000)
        ql.run()
#        print(type(vi.V),vi.V)
        value_array.append(ql.V[2])        
        print('QL discount: ',d, 'Policy: ',ql.policy)


#    iterations_array=[1,5,10,50,100,500,1000,5000,10000]
    iterations_array=[10000,15000,20000,25000,30000,35000,40000,45000,50000]
    value_array2=[]
    P,R = hiive.mdptoolbox.example.forest(S=3,r1=4,r2=2,p=0.1,is_sparse=False)
    print('')
    print('Optimal Policies')
    for i in iterations_array:
        ql=hiive.mdptoolbox.mdp.QLearning(P,R,0.5,n_iter=i)
        ql.run()
#        print(type(vi.V),vi.V)
        value_array2.append(ql.V[2])        
        print('QL Num Iters: ',i, 'Policy: ',ql.policy)


    #Chart 1: Value Function, S=3 for different discount factors
#    discount_array=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    plt.plot(discount_array,value_array)#,discount_array,entropy_base)#(range(1,number_components),silhouette_vector)
    plt.title('Q-Learning: Value of State 3, by Discount Factor (Gamma)')#'Information Gain by Cluster'
    plt.xlabel('Discount Factor - Gamma')
    plt.ylabel('Value')
#    plt.legend(['Clusters','Baseline'])
#    plt.ylim((0,1.0))
    plt.show()

    #Chart 1: Value Function, S=3 for different number of iterations
#    discount_array=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    plt.plot(iterations_array,value_array2)#,discount_array,entropy_base)#(range(1,number_components),silhouette_vector)
    plt.title('Q-Learning: Value of State 3, by Num Iterations')#'Information Gain by Cluster'
    plt.xlabel('Number of Iterations')
    plt.ylabel('Value')
#    plt.legend(['Clusters','Baseline'])
#    plt.ylim((0,1.0))
    plt.show()

    return


if __name__ == '__main__':
    main()


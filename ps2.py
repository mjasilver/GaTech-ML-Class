
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

def main():
#    queens()
#    queens_max() #configured correctly
    four_peaks() #configured correctly
    six_peaks() #configured correctly
###    one_max() #configured correctly
###    flip_flop() #configured correctly
#    max_k_color()
#    tsp_max()
    knapsack()   #configured correctly
#    df_submission,df_test,df_train=process_data()
#    neural_net(df_submission,df_test,df_train)
#Neural Net
    df_submission,df_test,df_train=process_data2()
    neural_net_RHC(df_submission,df_test,df_train)
    neural_net_SA(df_submission,df_test,df_train)
    neural_net_GA(df_submission,df_test,df_train)

def queens():
    #Implements mlrose on the Queens problem
    print("------ QUEENS ------")
    #Simulated Annealing
    fitness=mlrose.Queens()
    problem = mlrose.DiscreteOpt(length=8,fitness_fn=fitness,maximize=False,max_val=8)
    schedule=mlrose.ExpDecay()
    init_state=None#np.array([0,1,2,3,4,5,6,7])
    random_state=None#1
    best_state,best_fitness=mlrose.simulated_annealing(problem,schedule=schedule,
                                                       max_attempts=100, max_iters=1000,
                                                       init_state=init_state,random_state=random_state)
    print('SA, best state:',best_state)
    print('SA, best fitness:',best_fitness)

    #Random Hill Climb
#    fitness=mlrose.Queens()
#    problem = mlrose.DiscreteOpt(length=8,fitness_fn=fitness,maximize=False,max_val=8)
#    schedule=mlrose.ExpDecay()
    init_state=None#    init_state=np.array([0,1,2,3,4,5,6,7])
    random_state=None#1
    best_state,best_fitness,fitness_RHC=mlrose.random_hill_climb(problem,
                                                       max_attempts=100, max_iters=1000,
                                                       init_state=init_state,random_state=random_state
                                                       ,curve=True)
    print('RHC, best state:',best_state)
    print('RHC, best fitness:',best_fitness)
    print('RHC fitness curve:',fitness_RHC)

    #Genetic Algorithm
#    fitness=mlrose.Queens()
#    problem = mlrose.DiscreteOpt(length=8,fitness_fn=fitness,maximize=False,max_val=8)
#    schedule=mlrose.ExpDecay()
#    init_state=None#    init_state=np.array([0,1,2,3,4,5,6,7])
    random_state=None#1
    best_state,best_fitness=mlrose.random_hill_climb(problem,
                                                       max_attempts=100, max_iters=1000,
                                                       random_state=random_state)
    print('GA, best state:',best_state)
    print('GA, best fitness:',best_fitness)


    #MIMIC
#    fitness=mlrose.Queens()
#    problem = mlrose.DiscreteOpt(length=8,fitness_fn=fitness,maximize=False,max_val=8)
#    schedule=mlrose.ExpDecay()
#    init_state=None#    init_state=np.array([0,1,2,3,4,5,6,7])
    random_state=None#1
    best_state,best_fitness=mlrose.random_hill_climb(problem,
                                                       max_attempts=100, max_iters=1000,
                                                       random_state=random_state)
    print('MIMIC, best state:',best_state)
    print('MIMIC, best fitness:',best_fitness)
    return


def four_peaks():
    #Implements mlrose on the Queens problem
    print("------ FOUR PEAKS ------")
    print("Supposedly highlights MIMIC")
    t_pct=0.1
    fitness=mlrose.FourPeaks(t_pct=t_pct)
    length=60
    problem = mlrose.DiscreteOpt(length=length,fitness_fn=fitness,maximize=True,max_val=2)

#    random_state=np.random.randint(low=0,high=100)
    init_state=None#np.array([1,1,1,0,1,0,0,1,0,0,0,0])#None
    print('rando state',init_state)

    #Simulated Annealing
    schedule=mlrose.ExpDecay()
#    init_state=None#np.array([1,1,1,0,1,0,0,1,0,0,0,0])#None
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_SA=mlrose.simulated_annealing(problem,schedule=schedule,
                                                       max_attempts=20000, max_iters=20000,
                                                       init_state=init_state,random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_SA=end_time-start_time
    print('SA, best state:',best_state)
    print('SA, best fitness:',best_fitness)

    #Random Hill Climb
#    init_state=None#    init_state=np.array([0,1,2,3,4,5,6,7])
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_RHC=mlrose.random_hill_climb(problem,
                                                       max_attempts=20000, max_iters=20000,
                                                       init_state=init_state,random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_RHC=end_time-start_time
    print('RHC, best state:',best_state)
    print('RHC, best fitness:',best_fitness)


    #Genetic Algorithm
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_GA=mlrose.genetic_alg(problem,
                                                       max_attempts=20000, max_iters=20000,#20000
                                                       random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_GA=end_time-start_time
    print('GA, best state:',best_state)
    print('GA, best fitness:',best_fitness)


    #MIMIC
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_MIMIC=mlrose.mimic(problem,
                                                       max_attempts=20000, max_iters=20000,#20000
                                                       random_state=random_state,
                                                       curve=True, fast_mimic=True)
    end_time=time.time()
    time_MIMIC=end_time-start_time
    print('MIMIC, best state:',best_state)
    print('MIMIC, best fitness:',best_fitness)

    iters_MIMIC=np.arange(len(curve_MIMIC))
    iters_RHC=np.arange(len(curve_RHC))
    iters_SA=np.arange(len(curve_SA))
    iters_GA=np.arange(len(curve_GA))
#    print('iters_list',iters_list)
    plt.plot(iters_MIMIC,curve_MIMIC,iters_RHC,curve_RHC,iters_SA,curve_SA,iters_GA,curve_GA)
    plt.title('Performance by Iterations, Four Peaks')
    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    #plt.ylim((0,1.0))
    plt.savefig('charts/four_peaks_accuracy.png')
    plt.show()


    time_to_process=[time_MIMIC/20,time_SA/20,time_GA/20,time_RHC/20]
    algorithm_labels=['MIMIC','SA','GA','RHC']
    plt.bar(algorithm_labels,time_to_process,width=0.8,bottom=None,align='center')
    plt.title('Time per 1000 Iterations, Four Peaks')
#    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Algorithm')
    plt.ylabel('Time')
    #plt.ylim((0,1.0))
    plt.savefig('charts/four_peaks_iterationTime.png')
    plt.show()


    iters_MIMIC=np.arange(len(curve_MIMIC))*200
#    print('iters_MIMIC',iters_MIMIC)
    iters_RHC=np.arange(len(curve_RHC))
    iters_SA=np.arange(len(curve_SA))
    iters_GA=np.arange(len(curve_GA))*200
#    print('iters_list',iters_list)
    plt.plot(iters_MIMIC,curve_MIMIC,iters_RHC,curve_RHC,iters_SA,curve_SA,iters_GA,curve_GA)
    plt.title('Performance by Function Evaluations, Four Peaks')
    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Function Evaluations')
    plt.ylabel('Score')
    #plt.ylim((0,1.0))
    plt.xlim((0,20000.0))
    plt.savefig('charts/four_peaks_function_evals.png')
    plt.show()


    return




def six_peaks():
    #Implements mlrose on the Queens problem
    print("------ SIX PEAKS ------")
    print("Supposedly can highlight MIMIC")
    t_pct=0.15#0.2
    fitness=mlrose.SixPeaks(t_pct=t_pct)
    length=80#40
    problem = mlrose.DiscreteOpt(length=length,fitness_fn=fitness,maximize=True,max_val=2)

#    random_state=np.random.randint(low=0,high=100)
    init_state=None#np.array([1,1,1,0,1,0,0,1,0,0,0,0])#None
    print('rando state',init_state)

    #Simulated Annealing
    schedule=mlrose.ExpDecay()
#    init_state=None#np.array([1,1,1,0,1,0,0,1,0,0,0,0])#None
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_SA=mlrose.simulated_annealing(problem,schedule=schedule,
                                                       max_attempts=1000000, max_iters=1000000,
                                                       init_state=init_state,random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_SA=end_time-start_time
    print('SA, best state:',best_state)
    print('SA, best fitness:',best_fitness)

    #Random Hill Climb
#    init_state=None#    init_state=np.array([0,1,2,3,4,5,6,7])
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_RHC=mlrose.random_hill_climb(problem,
                                                       max_attempts=1000000, max_iters=1000000,
                                                       init_state=init_state,random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_RHC=end_time-start_time
    print('RHC, best state:',best_state)
    print('RHC, best fitness:',best_fitness)


    #Genetic Algorithm
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_GA=mlrose.genetic_alg(problem,pop_size=500,#200,
                                                       max_attempts=250, max_iters=100,
                                                       random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_GA=end_time-start_time
    print('GA, best state:',best_state)
    print('GA, best fitness:',best_fitness)


    #MIMIC
    '''
    random_state=None#1
    start_time=time.time()
    pop_list=[500,1000,2000] #pop size originally at 500
    iters_list=[100,200,500] #iters originally at 50 or something idk
    grid_results=np.zeros((len(pop_list),len(iters_list)))
    i=0
    while i < len(pop_list):
        pop=pop_list[i]
        print('population:',pop)
        j=0
        while j < len (iters_list):
            iters=iters_list[j]
            print('iters:',iters)
            best_state,best_fitness,curve_MIMIC=mlrose.mimic(problem,pop_size=pop,
                                                           keep_pct=0.3,
                                                       max_attempts=iters,
                                                           max_iters=iters,
                                                       random_state=random_state, curve=True)
            grid_results[i,j]=best_fitness
#            print('MIMIC, best state:',best_state)
            print('MIMIC, best fitness:',best_fitness)
            j=j+1
        i=i+1
    end_time=time.time()
    time_MIMIC=end_time-start_time
    print('MIMIC grid results:',grid_results)
    print('MIMIC best result:',np.amax(grid_results))
    '''

    random_state=None#1
    start_time=time.time()
    pop_list=[500] #pop size originally at 500
    iters_list=[100] #iters originally at 50 or something idk
    grid_results=np.zeros((len(pop_list),len(iters_list)))
    pop=pop_list[0]
    iters=iters_list[0]
    best_state,best_fitness,curve_MIMIC=mlrose.mimic(problem,pop_size=pop,
                                                           keep_pct=0.3,
                                                       max_attempts=iters,
                                                           max_iters=iters,
                                                       random_state=random_state, curve=True)
    print('MIMIC, best state:',best_state)
    print('MIMIC, best fitness:',best_fitness)
    end_time=time.time()
    time_MIMIC=end_time-start_time



    #CHARTS
    iters_MIMIC=np.arange(len(curve_MIMIC))
    iters_RHC=np.arange(len(curve_RHC))
    iters_SA=np.arange(len(curve_SA))
    iters_GA=np.arange(len(curve_GA))
#    print('iters_list',iters_list)
    plt.plot(iters_MIMIC,curve_MIMIC,iters_RHC,curve_RHC,iters_SA,curve_SA,iters_GA,curve_GA)
    plt.title('Performance by Iterations, Six Peaks')
    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    #plt.ylim((0,1.0))
    plt.xlim((0,1000.0))
    plt.savefig('charts/six_peaks_accuracy.png')
    plt.show()


    time_to_process=[time_MIMIC*4,time_SA/1000,time_GA*4,time_RHC/1000]
    algorithm_labels=['MIMIC','SA','GA','RHC']
    plt.bar(algorithm_labels,time_to_process,width=0.8,bottom=None,align='center')
    plt.title('Time per 1000 Iterations, Six Peaks')
#    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Algorithm')
    plt.ylabel('Time')
    #plt.ylim((0,1.0))
    plt.savefig('charts/six_peaks_iterationTime.png')
    plt.show()



    iters_MIMIC=np.arange(len(curve_MIMIC))*500
#    print('iters_MIMIC',iters_MIMIC)
    iters_RHC=np.arange(len(curve_RHC))
    iters_SA=np.arange(len(curve_SA))
    iters_GA=np.arange(len(curve_GA))*500
#    print('iters_list',iters_list)
    plt.plot(iters_MIMIC,curve_MIMIC,iters_RHC,curve_RHC,iters_SA,curve_SA,iters_GA,curve_GA)
    plt.title('Performance by Function Evaluations, Six Peaks')
    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Function Evaluations')
    plt.ylabel('Score')
    #plt.ylim((0,1.0))
    plt.xlim((0,50000.0))
    plt.savefig('charts/six_peaks_function_evals.png')
    plt.show()

    return


def one_max():
    #Implements mlrose on the Queens problem
    print("------ ONE MAX ------")
    print("To highlight Simulated Annealing???")
    fitness=mlrose.OneMax()
    length=12
    problem = mlrose.DiscreteOpt(length=length,fitness_fn=fitness,maximize=True,max_val=2)

#    random_state=np.random.randint(low=0,high=100)
    init_state=None#np.array([1,1,1,0,1,0,0,1,0,0,0,0])#None
    print('rando state',init_state)

    #Simulated Annealing
    schedule=mlrose.ExpDecay()
#    init_state=None#np.array([1,1,1,0,1,0,0,1,0,0,0,0])#None
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_SA=mlrose.simulated_annealing(problem,schedule=schedule,
                                                       max_attempts=2000, max_iters=2000,
                                                       init_state=init_state,random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_SA=end_time-start_time
    print('SA, best state:',best_state)
    print('SA, best fitness:',best_fitness)

    #Random Hill Climb
#    init_state=None#    init_state=np.array([0,1,2,3,4,5,6,7])
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_RHC=mlrose.random_hill_climb(problem,
                                                       max_attempts=2000, max_iters=2000,
                                                       init_state=init_state,random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_RHC=end_time-start_time
    print('RHC, best state:',best_state)
    print('RHC, best fitness:',best_fitness)


    #Genetic Algorithm
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_GA=mlrose.genetic_alg(problem,
                                                       max_attempts=2000, max_iters=2000,
                                                       random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_GA=end_time-start_time
    print('GA, best state:',best_state)
    print('GA, best fitness:',best_fitness)


    #MIMIC
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_MIMIC=mlrose.mimic(problem,
                                                       max_attempts=2000, max_iters=2000,
                                                       random_state=random_state,
                                                       curve=True, fast_mimic=True)
    end_time=time.time()
    time_MIMIC=end_time-start_time
    print('MIMIC, best state:',best_state)
    print('MIMIC, best fitness:',best_fitness)

    iters_MIMIC=np.arange(len(curve_MIMIC))
    iters_RHC=np.arange(len(curve_RHC))
    iters_SA=np.arange(len(curve_SA))
    iters_GA=np.arange(len(curve_GA))
#    print('iters_list',iters_list)
    plt.plot(iters_MIMIC,curve_MIMIC,iters_RHC,curve_RHC,iters_SA,curve_SA,iters_GA,curve_GA)
    plt.title('Performance by Iterations, One Max')
    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    #plt.ylim((0,1.0))
    plt.savefig('charts/one_max_accuracy.png')
    plt.show()


    time_to_process=[time_MIMIC,time_SA,time_GA,time_RHC]
    algorithm_labels=['MIMIC','SA','GA','RHC']
    plt.bar(algorithm_labels,time_to_process,width=0.8,bottom=None,align='center')
    plt.title('Time per 100 Iterations, One Max')
#    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Algorithm')
    plt.ylabel('Time')
    #plt.ylim((0,1.0))
    plt.savefig('charts/one_max_iterationTime.png')
    plt.show()

    return



def flip_flop():
    #Implements mlrose on the Queens problem
    print("------ Flip Flop ------")
    print("To highlight GA??? -- Highlight MIMIC instead bc its one-node dependency???")
    fitness=mlrose.FlipFlop()
    length=25
    problem = mlrose.DiscreteOpt(length=length,fitness_fn=fitness,maximize=True,max_val=2)

#    random_state=np.random.randint(low=0,high=100)
    init_state=None#np.array([1,1,1,0,1,0,0,1,0,0,0,0])#None
    print('rando state',init_state)

    #Simulated Annealing
    schedule=mlrose.ExpDecay()
#    init_state=None#np.array([1,1,1,0,1,0,0,1,0,0,0,0])#None
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_SA=mlrose.simulated_annealing(problem,schedule=schedule,
                                                       max_attempts=500, max_iters=500,
                                                       init_state=init_state,random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_SA=end_time-start_time
    print('SA, best state:',best_state)
    print('SA, best fitness:',best_fitness)

    #Random Hill Climb
#    init_state=None#    init_state=np.array([0,1,2,3,4,5,6,7])
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_RHC=mlrose.random_hill_climb(problem,
                                                       max_attempts=500, max_iters=500,
                                                       init_state=init_state,random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_RHC=end_time-start_time
    print('RHC, best state:',best_state)
    print('RHC, best fitness:',best_fitness)


    #Genetic Algorithm
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_GA=mlrose.genetic_alg(problem,
                                                       max_attempts=500, max_iters=500,
                                                       random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_GA=end_time-start_time
    print('GA, best state:',best_state)
    print('GA, best fitness:',best_fitness)


    #MIMIC
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_MIMIC=mlrose.mimic(problem,
                                                       max_attempts=500, max_iters=500,
                                                       random_state=random_state,
                                                       curve=True, fast_mimic=True)
    end_time=time.time()
    time_MIMIC=end_time-start_time
    print('MIMIC, best state:',best_state)
    print('MIMIC, best fitness:',best_fitness)

    iters_MIMIC=np.arange(len(curve_MIMIC))
    iters_RHC=np.arange(len(curve_RHC))
    iters_SA=np.arange(len(curve_SA))
    iters_GA=np.arange(len(curve_GA))
#    print('iters_list',iters_list)
    plt.plot(iters_MIMIC,curve_MIMIC,iters_RHC,curve_RHC,iters_SA,curve_SA,iters_GA,curve_GA)
    plt.title('Performance by Iterations, Flip Flop')
    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    #plt.ylim((0,1.0))
    plt.savefig('charts/flip_flop_accuracy.png')
    plt.show()


    time_to_process=[time_MIMIC,time_SA,time_GA,time_RHC]
    algorithm_labels=['MIMIC','SA','GA','RHC']
    plt.bar(algorithm_labels,time_to_process,width=0.8,bottom=None,align='center')
    plt.title('Time per 100 Iterations, Flip Flop')
#    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Algorithm')
    plt.ylabel('Time')
    #plt.ylim((0,1.0))
    plt.savefig('charts/flip_flop_iterationTime.png')
    plt.show()

    return

def queens_max():
    #Implements mlrose on the Queens problem
    print("------ QUEENS MAX ------")
    print("PAIRS of non-attacking queens -- max is 28 for length 8")
#    print("Highlights Simulated Annealing?")
#    fitness=mlrose.Queens()
    fitness_custom = mlrose.CustomFitness(queens_max_fitness)
    problem = mlrose.DiscreteOpt(length=15,fitness_fn=fitness_custom,maximize=True,max_val=15)
    schedule=mlrose.ExpDecay()


    #Simulated Annealing
    init_state=None#np.array([4,1,3,5,7,2,0,6])#None
    random_state=None#1#None
    start_time=time.time()
    best_state,best_fitness,curve_SA=mlrose.simulated_annealing(problem,schedule=schedule,
                                                       max_attempts=1000, max_iters=1000,
                                                       init_state=init_state,random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_SA=end_time-start_time
    print('SA, best state:',best_state)
    print('SA, best fitness:',best_fitness)

    #Random Hill Climb
    init_state=None#    init_state=np.array([0,1,2,3,4,5,6,7])
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_RHC=mlrose.random_hill_climb(problem,
                                                       max_attempts=1000, max_iters=1000,
                                                       init_state=init_state,random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_RHC=end_time-start_time
    print('RHC, best state:',best_state)
    print('RHC, best fitness:',best_fitness)
#    print('RHC fitness curve:',fitness_RHC)


    #Genetic Algorithm
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_GA=mlrose.genetic_alg(problem,
                                                       max_attempts=1000, max_iters=1000,
                                                       random_state=random_state,curve=True)
    end_time=time.time()
    time_GA=end_time-start_time
    print('GA, best state:',best_state)
    print('GA, best fitness:',best_fitness)


    #MIMIC
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_MIMIC=mlrose.mimic(problem,
                                                       max_attempts=1000, max_iters=1000,
                                                       random_state=random_state,curve=True)
    end_time=time.time()
    time_MIMIC=end_time-start_time
    print('MIMIC, best state:',best_state)
    print('MIMIC, best fitness:',best_fitness)



    iters_MIMIC=np.arange(len(curve_MIMIC))
    iters_RHC=np.arange(len(curve_RHC))
    iters_SA=np.arange(len(curve_SA))
    iters_GA=np.arange(len(curve_GA))
#    print('iters_list',iters_list)
    plt.plot(iters_MIMIC,curve_MIMIC,iters_RHC,curve_RHC,iters_SA,curve_SA,iters_GA,curve_GA)
    plt.title('Performance by Iterations, Flip Flop')
    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    #plt.ylim((0,1.0))
    plt.savefig('charts/queens_max_accuracy.png')
    plt.show()


    time_to_process=[time_MIMIC,time_SA,time_GA,time_RHC]
    algorithm_labels=['MIMIC','SA','GA','RHC']
    plt.bar(algorithm_labels,time_to_process,width=0.8,bottom=None,align='center')
    plt.title('Time per 100 Iterations, Queens Max')
#    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Algorithm')
    plt.ylabel('Time')
    #plt.ylim((0,1.0))
    plt.savefig('charts/queens_max_iterationTime.png')
    plt.show()


    return


def queens_max_fitness(state):

   # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
    # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) and (state[j] != state[i] + (j - i)) and (state[j] != state[i] - (j - i)):
            # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt

def queens_max_fitness2(state):
   # print("new count")
   # Initialize counter
    fitness_cnt = 0

          # For all pairs of queens
    for i in range(len(state) - 1):
        attacks=0
        for j in range(i + 1, len(state)):
        # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) and (state[j] != state[i] + (j - i)) and (state[j] != state[i] - (j - i)):
                #Condition for no attacks
                attacks = attacks
            else:
                attacks += 1
        if attacks==0:
            # If no attacks, then increment counter
            fitness_cnt += 1

    return fitness_cnt



def knapsack():
    #Implements mlrose on the Queens problem
    print("------ KNAPSACK ------")
    print("Supposedly highlights --> Can use GA here, possibly MIMIC bc it has Fast Mimic while TSP does not?????")
    #weights = [10, 5, 2, 8, 15]
    #values = [1, 2, 3, 4, 5]
    weights = [10, 5, 2, 8, 15,4,8,9,7,12]
    values = [1, 2, 3, 4, 5,6,7,8,9,10]
    max_weight_pct = 1.0
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    length=10
    problem = mlrose.DiscreteOpt(length=length,fitness_fn=fitness,maximize=True,max_val=4)



    #Simulated Annealing
    schedule=mlrose.ExpDecay()
    init_state=None#np.array([1,1,1,0,1,0,0,1,0,0,0,0])#None
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_SA=mlrose.simulated_annealing(problem,schedule=schedule,
                                                       max_attempts=200000, max_iters=200000,#1000
                                                       init_state=init_state,random_state=random_state,curve=True)
    end_time=time.time()
    time_SA=end_time-start_time
    print('SA, best state:',best_state)
    print('SA, best fitness:',best_fitness)

    #Random Hill Climb
    init_state=None#    init_state=np.array([0,1,2,3,4,5,6,7])
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_RHC=mlrose.random_hill_climb(problem,
                                                       max_attempts=200000, max_iters=200000,#1000
                                                       init_state=init_state,random_state=random_state,curve=True)
    end_time=time.time()
    time_RHC=end_time-start_time
    print('RHC, best state:',best_state)
    print('RHC, best fitness:',best_fitness)


    #Genetic Algorithm
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_GA=mlrose.genetic_alg(problem,
                                                       max_attempts=1000, max_iters=1000,
                                                       random_state=random_state,curve=True)
    end_time=time.time()
    time_GA=end_time-start_time
    print('GA, best state:',best_state)
    print('GA, best fitness:',best_fitness)


    #MIMIC
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_MIMIC=mlrose.mimic(problem,
                                                       max_attempts=1000, max_iters=1000,
                                                       random_state=random_state,curve=True)
    end_time=time.time()
    time_MIMIC=end_time-start_time
    print('MIMIC, best state:',best_state)
    print('MIMIC, best fitness:',best_fitness)


    iters_MIMIC=np.arange(len(curve_MIMIC))
    iters_RHC=np.arange(len(curve_RHC))
    iters_SA=np.arange(len(curve_SA))
    iters_GA=np.arange(len(curve_GA))
#    print('iters_list',iters_list)
    plt.plot(iters_MIMIC,curve_MIMIC,iters_RHC,curve_RHC,iters_SA,curve_SA,iters_GA,curve_GA)
    plt.title('Performance by Iterations, Knapsack')
    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    #plt.ylim((0,1.0))
    plt.xlim((0,1000.0))
    plt.savefig('charts/knapsack_max_accuracy.png')
    plt.show()


    time_to_process=[time_MIMIC,time_SA/200.0,time_GA,time_RHC/200.0]
    algorithm_labels=['MIMIC','SA','GA','RHC']
    plt.bar(algorithm_labels,time_to_process,width=0.8,bottom=None,align='center')
    plt.title('Time per 1000 Iterations, Knapsack')
#    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Algorithm')
    plt.ylabel('Time')
    #plt.ylim((0,1.0))
    plt.savefig('charts/knapsack_max_iterationTime.png')
    plt.show()


    iters_MIMIC=np.arange(len(curve_MIMIC))*200
#    print('iters_MIMIC',iters_MIMIC)
    iters_RHC=np.arange(len(curve_RHC))
    iters_SA=np.arange(len(curve_SA))
    iters_GA=np.arange(len(curve_GA))*200
#    print('iters_list',iters_list)
    plt.plot(iters_MIMIC,curve_MIMIC,iters_RHC,curve_RHC,iters_SA,curve_SA,iters_GA,curve_GA)
    plt.title('Performance by Function Evaluations, Six Peaks')
    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Function Evaluations')
    plt.ylabel('Score')
    #plt.ylim((0,1.0))
    plt.xlim((0,200000.0))
    plt.savefig('charts/knapsack_function_evals.png')
    plt.show()


    return

def max_k_color():
    #Implements mlrose on the Queens problem
    print("------ MAX K COLOR ------")
    print("Supposedly highlights --> Can use MIMIC here????? --> Follow Isbell's paper for this one")
    edges = [(0, 1), (0, 2),(0,3), (0, 4),(0,5),(0,6),(0,7),(0,8)
             ,(1, 0), (1, 2),(1,3), (1, 4),(1,5),(1,6),(1,7),(1,8)
             ,(2, 0), (2, 1),(2,3), (2, 4),(2,5),(2,6),(2,7),(2,8)
             ,(3, 0), (3, 1),(3,2), (3, 4),(3,5),(3,6),(3,7),(3,8)
             ,(4, 0), (4, 1),(4,2), (4, 3),(4,5),(4,6),(4,7),(4,8)
             ,(5, 0), (5, 1),(5,2), (5, 4),(5,3),(5,6),(5,7),(5,8)
             ,(6, 0), (6, 1),(6,2), (6, 4),(6,5),(6,3),(6,7),(6,8)
             ,(7, 0), (7, 1),(7,2), (7, 4),(7,5),(7,6),(7,3),(7,8)
             ,(8, 0), (8, 1),(8,2), (8, 4),(8,5),(8,6),(8,7),(8,3)
             ]
    fitness = mlrose.MaxKColor(edges)
    length=9
    problem = mlrose.DiscreteOpt(length=length,fitness_fn=fitness,maximize=True,max_val=5)
#    fitness_custom = mlrose.CustomFitness(max_k_custom_fitness)
#    problem = mlrose.DiscreteOpt(length=9,fitness_fn=fitness_custom,maximize=False,max_val=4)

    #Simulated Annealing
    schedule=mlrose.ExpDecay()
    init_state=None#np.array([1,1,1,0,1,0,0,1,0,0,0,0])#None
    random_state=None#1
    best_state,best_fitness=mlrose.simulated_annealing(problem,schedule=schedule,
                                                       max_attempts=1000, max_iters=1000,
                                                       init_state=init_state,random_state=random_state)
    print('SA, best state:',best_state)
    print('SA, best fitness:',best_fitness)

    #Random Hill Climb
    init_state=None#    init_state=np.array([0,1,2,3,4,5,6,7])
    random_state=None#1
    best_state,best_fitness=mlrose.random_hill_climb(problem,
                                                       max_attempts=1000, max_iters=1000,
                                                       init_state=init_state,random_state=random_state)
    print('RHC, best state:',best_state)
    print('RHC, best fitness:',best_fitness)


    #Genetic Algorithm
    random_state=None#1
    best_state,best_fitness=mlrose.genetic_alg(problem,
                                                       max_attempts=1000, max_iters=1000,
                                                       random_state=random_state)
    print('GA, best state:',best_state)
    print('GA, best fitness:',best_fitness)


    #MIMIC
    random_state=None#1
    best_state,best_fitness=mlrose.mimic(problem,
                                                       max_attempts=1000, max_iters=1000,
                                                       random_state=random_state, fast_mimic=True)
    print('MIMIC, best state:',best_state)
    print('MIMIC, best fitness:',best_fitness)

    return



def tsp_max():
    #Implements mlrose on the Queens problem
    print("------ TRAVELLING SALESMAN MAX ------")
    print("Highlighting MIMIC? Can Highlight GA if necessary")
    
    coords_list = [(1, 1), (4, 2), (5, 2)
                   , (6, 4), (4, 4), (7, 6), (1, 5), (2, 3)
                    ,(8, 10), (4, 8), (3, 2), (6, 3)
                   , (3, 8), (2, 6), (11, 4), (1, 9)
                   , (9, 9), (7, 6), (10, 9), (10, 5)
                   ]
    '''
    coords_list = [(1, 1), (1, 2), (1, 3),(1,4),(1,5),(1,6),(1,7)
                   , (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2,6),(2,7)
                    ,(3, 1), (3, 2), (3, 3), (3, 4),(3,5),(3,6),(3,7)
                   , (4, 1), (4, 2), (4, 3), (4, 4),(4,5),(4,6),(4,7)
                   , (5, 1), (5, 2), (5, 3), (5, 4),(5,5),(5,6),(5,7)
                   , (6, 1), (6, 2), (6, 3), (6, 4),(6,5),(6,6),(6,7)
                   , (7, 1), (7, 2), (7, 3), (7, 4),(7,5),(7,6),(7,7)
                   ]
    '''
#    dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), \
#             (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), \
#             (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426), \
#             (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721), \
#             (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), \
#             (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), \
#             (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]
    #fitness_custom = mlrose.CustomFitness(tsp_fitness_max)
    #problem = mlrose.DiscreteOpt(length=15,fitness_fn=fitness_custom,maximize=True,max_val=15)
#    fitness=mlrose.TravellingSales(coords=coords_list)


#Trying out custom fitness function
    
    fitness_dists=mlrh.TravellingSales(coords=coords_list)#distances=dist_list)
#    problem = mlrose.DiscreteOpt(length=8,fitness_fn=fitness,maximize=False)
    problem = mlrh.TSPOpt(length=20,fitness_fn=fitness_dists,maximize=False)
    schedule=mlrh.ExpDecay()
    '''
    fitness=mlrh.CustomFitness(tsp_fitness)
    problem=mlrh.TSPOpt(length=12,fitness_fn=fitness,maximize=True)
    '''

    #Simulated Annealing
    init_state=None#np.array([4,1,3,5,7,2,0,6])#None
    random_state=None#1#None
    #eval_count_SA=0
    #eval_count=0
    start_time=time.time()
    best_state,best_fitness,curve_SA=mlrh.simulated_annealing(problem,schedule=schedule,
                                                       max_attempts=10000, max_iters=10000,
                                                       init_state=init_state,random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_SA=end_time-start_time
    print('SA, best state:',best_state)
    print('SA, best fitness:',best_fitness)
    #eval_count_SA=eval_count
    #print('SA Eval Count',eval_count_SA)
    #Random Hill Climb
    init_state=None#    init_state=np.array([0,1,2,3,4,5,6,7])
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_RHC=mlrh.random_hill_climb(problem,
                                                       max_attempts=10000, max_iters=10000,
                                                       init_state=init_state,random_state=random_state,
                                                       curve=True)
    end_time=time.time()
    time_RHC=end_time-start_time
    print('RHC, best state:',best_state)
    print('RHC, best fitness:',best_fitness)
#    print('RHC fitness curve:',fitness_RHC)


    #Genetic Algorithm
    random_state=None#1
    start_time=time.time()
    best_state,best_fitness,curve_GA=mlrh.genetic_alg(problem,pop_size=50,
                                                       max_attempts=2000, max_iters=2000,
                                                       random_state=random_state,curve=True)
    end_time=time.time()
    time_GA=end_time-start_time
    print('GA, best state:',best_state)
    print('GA, best fitness:',best_fitness)


    #MIMIC
    random_state=None#1
    start_time=time.time()
    grid_results=np.zeros((3,2))
    pop_list=[500,1000] #pop size originally at 500
    iters_list=[50,100] #iters originally at 50 or something idk
    i=0
    while i < len(pop_list):
        pop=pop_list[i]
        print('population:',pop)
        j=0
        while j < len (iters_list):
            iters=iters_list[j]
            print('iters:',iters)
            best_state,best_fitness,curve_MIMIC=mlrh.mimic(problem,pop_size=pop,
                                                           keep_pct=0.3,
                                                       max_attempts=iters,
                                                           max_iters=iters,
                                                       random_state=random_state, curve=True)
            grid_results[i,j]=best_fitness
#            print('MIMIC, best state:',best_state)
            print('MIMIC, best fitness:',best_fitness)
            j=j+1
        i=i+1
    end_time=time.time()
    time_MIMIC=end_time-start_time
    print('MIMIC grid results:',grid_results)
    print('MIMIC best result:',np.amin(grid_results))



    iters_MIMIC=np.arange(len(curve_MIMIC))
    iters_RHC=np.arange(len(curve_RHC))
    iters_SA=np.arange(len(curve_SA))
    iters_GA=np.arange(len(curve_GA))
#    print('iters_list',iters_list)
    plt.plot(iters_MIMIC,curve_MIMIC,iters_RHC,curve_RHC,iters_SA,curve_SA,iters_GA,curve_GA)
    plt.title('Performance by Iterations, Travelling Salesman')
    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    #plt.ylim((0,1.0))
    plt.savefig('charts/tsp_max_accuracy.png')
    plt.show()


    time_to_process=[time_MIMIC,time_SA,time_GA,time_RHC]
    algorithm_labels=['MIMIC','SA','GA','RHC']
    plt.bar(algorithm_labels,time_to_process,width=0.8,bottom=None,align='center')
    plt.title('Time per 100 Iterations, Traveling Salesman')
#    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Algorithm')
    plt.ylabel('Time')
    #plt.ylim((0,1.0))
    plt.savefig('charts/tsp_max_iterationTime.png')
    plt.show()
    return



def tsp_fitness(state):
    """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation. Each integer between 0 and
            (len(state) - 1), inclusive must appear exactly once in the array.

        Returns
        -------
        fitness: float
            Value of fitness function. Returns :code:`np.inf` if travel between
            two consecutive nodes on the tour is not possible.
    """
    coords_list = [(1, 1), (4, 2), (5, 2)
                   , (6, 4), (4, 4), (7, 6), (1, 5), (2, 3)
                    ,(8, 10), (4, 8), (3, 2), (6, 3)
#                   , (14, 14), (23, 16), (11, 5), (32, 33)
                   ]

    global eval_count
    fitness=mlrh.TravellingSales(coords=coords_list)#distances=dist_list)
    eval_count += 1

    return fitness.evaluate(state)



def max_k_custom_fitness(state):
    """
        Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
    """
    fitness = 0
    for i in range(len(self.edges)):
        # Check for adjacent nodes of the same color
        if state[self.edges[i][0]] == state[self.edges[i][1]]:
            fitness += 1

    return fitness




def process_data():
    print(os.getcwd())
    print(os.listdir("Titanic"))

#    df_results=pd.read_csv('horse_race_data/results.csv')
    df_submission=pd.read_csv('Titanic/gender_submission.csv')
    df_test=pd.read_csv('Titanic/test.csv')
    df_train=pd.read_csv('Titanic/train.csv')
    
    print('Columns of submission.csv:')
    print(df_submission.columns.values)

#    print('Columns of test.csv:')
#    print(df_test.columns.values)

#    print('Columns of train.csv:')
#    print(df_train.columns.values)

    #DEAL WITH N/A VALUES
#    print(df_test.query("PassengerId=='902'"))
    mean_age=df_train['Age'].mean(axis=0,skipna=True)
    mean_fare=df_train['Fare'].mean(axis=0,skipna=True)
    df_train['Age'].fillna(value=mean_age,inplace=True)
    df_train['Fare'].fillna(value=mean_fare,inplace=True)

    df_test['Age'].fillna(value=mean_age,inplace=True)
    df_test['Fare'].fillna(value=mean_fare,inplace=True)
#    print(df_test.query("PassengerId=='902'"))

    #CATEGORICAL VARIABLES
    ''' #Creating categorical type, but this doesn't seem to work with NNs
    df_train['Sex_cat']=df_train['Sex'].astype('category')
    df_train['Embarked_cat']=df_train['Embarked'].astype('category')
    df_train['Pclass_cat']=df_train['Pclass'].astype('category')

    df_test['Sex_cat']=df_test['Sex'].astype('category')
    df_test['Embarked_cat']=df_test['Embarked'].astype('category')
    df_test['Pclass_cat']=df_test['Pclass'].astype('category')
    '''

    '''
    #Creating Dummy Variables instead
    #https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
    unique_sex=df_train['Sex'].unique()
    unique_embarked=df_train['Embarked'].unique()
    unique_Pclass=df_train['Pclass'].unique()
    print('uniques',unique_sex,unique_embarked,unique_Pclass)
    #TRAIN SET
    #SEX
    df_train['Is_Male']=0
    df_train.loc[df_train['Sex']=='male','Is_Male']=1
    #EMBARKED
    df_train['Is_Embarked_S']=0
    df_train.loc[df_train['Embarked']=='S','Is_Embarked_S']=1
    df_train['Is_Embarked_C']=0
    df_train.loc[df_train['Embarked']=='C','Is_Embarked_C']=1
    df_train['Is_Embarked_Q']=0
    df_train.loc[df_train['Embarked']=='Q','Is_Embarked_Q']=1
    #PCLASS
    df_train['Is_Pclass_1']=0
    df_train['Is_Pclass_2']=0
    df_train['Is_Pclass_3']=0
    df_train.loc[df_train['Pclass']==1,'Is_Pclass_1']=1
    df_train.loc[df_train['Pclass']==2,'Is_Pclass_2']=1
    df_train.loc[df_train['Pclass']==3,'Is_Pclass_3']=1
    
    #TEST SET
    #SEX
    df_test['Is_Male']=0
    df_test.loc[df_test['Sex']=='male','Is_Male']=1
    #EMBARKED
    df_test['Is_Embarked_S']=0
    df_test.loc[df_test['Embarked']=='S','Is_Embarked_S']=1
    df_test['Is_Embarked_C']=0
    df_test.loc[df_test['Embarked']=='C','Is_Embarked_C']=1
    df_test['Is_Embarked_Q']=0
    df_test.loc[df_test['Embarked']=='Q','Is_Embarked_Q']=1
    #PCLASS
    df_test['Is_Pclass_1']=0
    df_test['Is_Pclass_2']=0
    df_test['Is_Pclass_3']=0
    df_test.loc[df_test['Pclass']==1,'Is_Pclass_1']=1
    df_test.loc[df_test['Pclass']==2,'Is_Pclass_2']=1
    df_test.loc[df_test['Pclass']==3,'Is_Pclass_3']=1
    print(df_train.head())

#    enc = preprocessing.OneHotEncoder(categories=[unique_sex,unique_embarked,unique_Pclass])
#    print(df_train.head())

    #THEN, DATASET IS READY!!!!!
    '''
    
    return df_submission,df_train,df_test


def neural_net(df_sub,df_train,df_test):
    print("------ NEURAL NET ------")
    from sklearn.datasets import load_iris

    # Load the Iris dataset
#    data = load_iris()
    df_submission=df_sub
    df_train=df_train
    df_test=df_test

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

    # Split data into training and test sets
#    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, \
#                                                    test_size = 0.2, random_state = 3)
    drop_list=['Survived','Name','Sex','Ticket','Cabin','PassengerId','Embarked']
    drop_list2=['Name','Sex','Ticket','Cabin','PassengerId','Embarked']

    X_train=df_train.drop(drop_list,axis=1)
    X_test=df_test.drop(drop_list2,axis=1)
    y_train=df_train['Survived']
    y_test=df_submission['Survived']

    # Normalize feature data
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One hot encode target values
    one_hot = OneHotEncoder()

    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 random_state = 3)

    nn_model1.fit(X_train_scaled, y_train_hot)
#    nn_model1.fit(X_train_scaled, y_train)

    from sklearn.metrics import accuracy_score

    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
#    y_train_accuracy = accuracy_score(y_train, y_train)

    print(y_train_accuracy)
    #0.45

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
#    y_test_accuracy = accuracy_score(y_test, y_test)

    print(y_test_accuracy)
    return



def process_data2():
    print(os.getcwd())
    print(os.listdir("Titanic"))

#    df_results=pd.read_csv('horse_race_data/results.csv')
    df_submission=pd.read_csv('Titanic/gender_submission.csv')
    df_test=pd.read_csv('Titanic/test.csv')
    df_train=pd.read_csv('Titanic/train.csv')
    
    print('Columns of submission.csv:')
    print(df_submission.columns.values)

    print('Columns of test.csv:')
    print(df_test.columns.values)

    print('Columns of train.csv:')
    print(df_train.columns.values)

    #DEAL WITH N/A VALUES
#    print(df_test.query("PassengerId=='902'"))
    mean_age=df_train['Age'].mean(axis=0,skipna=True)
    mean_fare=df_train['Fare'].mean(axis=0,skipna=True)
    df_train['Age'].fillna(value=mean_age,inplace=True)
    df_train['Fare'].fillna(value=mean_fare,inplace=True)

    df_test['Age'].fillna(value=mean_age,inplace=True)
    df_test['Fare'].fillna(value=mean_fare,inplace=True)
#    print(df_test.query("PassengerId=='902'"))

    #CATEGORICAL VARIABLES
    ''' #Creating categorical type, but this doesn't seem to work with NNs
    df_train['Sex_cat']=df_train['Sex'].astype('category')
    df_train['Embarked_cat']=df_train['Embarked'].astype('category')
    df_train['Pclass_cat']=df_train['Pclass'].astype('category')

    df_test['Sex_cat']=df_test['Sex'].astype('category')
    df_test['Embarked_cat']=df_test['Embarked'].astype('category')
    df_test['Pclass_cat']=df_test['Pclass'].astype('category')
    '''
    #Creating Dummy Variables instead
    #https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
    unique_sex=df_train['Sex'].unique()
    unique_embarked=df_train['Embarked'].unique()
    unique_Pclass=df_train['Pclass'].unique()
    print('uniques',unique_sex,unique_embarked,unique_Pclass)
    #TRAIN SET
    #SEX
    df_train['Is_Male']=0
    df_train.loc[df_train['Sex']=='male','Is_Male']=1
    #EMBARKED
    df_train['Is_Embarked_S']=0
    df_train.loc[df_train['Embarked']=='S','Is_Embarked_S']=1
    df_train['Is_Embarked_C']=0
    df_train.loc[df_train['Embarked']=='C','Is_Embarked_C']=1
    df_train['Is_Embarked_Q']=0
    df_train.loc[df_train['Embarked']=='Q','Is_Embarked_Q']=1
    #PCLASS
    df_train['Is_Pclass_1']=0
    df_train['Is_Pclass_2']=0
    df_train['Is_Pclass_3']=0
    df_train.loc[df_train['Pclass']==1,'Is_Pclass_1']=1
    df_train.loc[df_train['Pclass']==2,'Is_Pclass_2']=1
    df_train.loc[df_train['Pclass']==3,'Is_Pclass_3']=1
    
    #TEST SET
    #SEX
    df_test['Is_Male']=0
    df_test.loc[df_test['Sex']=='male','Is_Male']=1
    #EMBARKED
    df_test['Is_Embarked_S']=0
    df_test.loc[df_test['Embarked']=='S','Is_Embarked_S']=1
    df_test['Is_Embarked_C']=0
    df_test.loc[df_test['Embarked']=='C','Is_Embarked_C']=1
    df_test['Is_Embarked_Q']=0
    df_test.loc[df_test['Embarked']=='Q','Is_Embarked_Q']=1
    #PCLASS
    df_test['Is_Pclass_1']=0
    df_test['Is_Pclass_2']=0
    df_test['Is_Pclass_3']=0
    df_test.loc[df_test['Pclass']==1,'Is_Pclass_1']=1
    df_test.loc[df_test['Pclass']==2,'Is_Pclass_2']=1
    df_test.loc[df_test['Pclass']==3,'Is_Pclass_3']=1
    print(df_train.head())

#    enc = preprocessing.OneHotEncoder(categories=[unique_sex,unique_embarked,unique_Pclass])
#    print(df_train.head())

    #THEN, DATASET IS READY!!!!!

    
    return df_submission,df_train,df_test

def neural_net_RHC(df_sub,df_train,df_test):
    print("------ NEURAL NET, Random Hill Climb ------")
    from sklearn.datasets import load_iris

    # Load the Iris dataset
#    data = load_iris()
    df_submission=df_sub
    df_train=df_train
    df_test=df_test

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

    # Split data into training and test sets
#    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, \
#                                                    test_size = 0.2, random_state = 3)
    drop_list=['Survived','Name','Sex','Ticket','Cabin','PassengerId','Embarked']
    drop_list2=['Name','Sex','Ticket','Cabin','PassengerId','Embarked']

    X_train=df_train.drop(drop_list,axis=1)
    X_test=df_test.drop(drop_list2,axis=1)
    y_train=df_train['Survived']
    y_test=df_submission['Survived']

    # Normalize feature data
#    scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
#    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
#    X_test_scaled=scaler.transform(X_test)

    # One hot encode target values
    one_hot = OneHotEncoder()

    y_train_hot=y_train
    y_test_hot=y_test
#    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
#    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

    #PS1 MLPClassifier
    print('PS1 Accuracy')
    start_time=time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', max_iter=3000,learning_rate_init=0.001, solver='adam')
    mlp.fit(X_train_scaled,y_train_hot)
    end_time=time.time()
    time_BackProp=end_time-start_time
    print('TIME BACKPROP',time_BackProp, end_time,start_time)
    predictions = mlp.predict(X_test_scaled)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    print(pd.crosstab(y_test,predictions,rownames=['True'],colnames=['Predicted'],margins=True))    


    #PS2 Classifier
    print('PS2 Accuracy')


    #GRIDSEARCH
    '''
    print('MLROSE GRIDSEARCH')
    parameters = { 'hidden_nodes':[[8]],#[8],[4],[2]],
                   'activation':['relu'],
                   'max_iters':[3000],
                   'learning_rate_init':[0.001],#[0.0001,0.0005,0.001],
                   'algorithm':['gradient_descent']
        }

#    mlp=MLPClassifier()
#    clf=GridSearchCV(mlp,parameters)
    mlrose_gridsearch=mlrose.NeuralNetwork()
    clf=GridSearchCV(mlrose_gridsearch,parameters)
    print('clf',clf)
    clf.fit(X_train,y_train)
    sorted(clf.cv_results_.keys())
    print(clf.cv_results_.keys())
    print('best estimator',clf.best_estimator_)
    print('best score',clf.best_score_)
    raise NotImplementedError
    '''           

    start_time=time.time()
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 3000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = False, clip_max = 5, max_attempts = 100, \
                                 random_state = None,restarts=10)#3)

    nn_model1.fit(X_train_scaled, y_train_hot)
    end_time=time.time()
    time_RHC=end_time-start_time
    print('TIME BACKPROP',time_RHC, end_time,start_time)
#    nn_model1.fit(X_train_scaled, y_train)

    from sklearn.metrics import accuracy_score

    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
#    y_train_accuracy = accuracy_score(y_train, y_train)

    print('training accuracy',y_train_accuracy)
    #0.45

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
#    y_test_accuracy = accuracy_score(y_test, y_test)

    print('testing accuracy',y_test_accuracy)

    print(confusion_matrix(y_test_hot,y_test_pred))
    print(classification_report(y_test_hot,y_test_pred))
#    print(pd.crosstab(y_test_hot,y_test_pred,rownames=['True'],colnames=['Predicted'],margins=True))    


    time_to_process=[time_RHC+0.1,time_BackProp+0.1]
    algorithm_labels=['RHC','Backprop']
    plt.bar(algorithm_labels,time_to_process,width=0.8,bottom=None,align='center')
    plt.title('Time per 3000 Iterations, Neural Network Training')
#    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Algorithm')
    plt.ylabel('Time')
#    plt.ylim((0,10.0))
    plt.savefig('charts/NN_RHC_iterationTime.png')
    plt.show()


    return


def neural_net_SA(df_sub,df_train,df_test):
    print("------ NEURAL NET, Simulated Annealing ------")
    from sklearn.datasets import load_iris

    # Load the Iris dataset
#    data = load_iris()
    df_submission=df_sub
    df_train=df_train
    df_test=df_test

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

    # Split data into training and test sets
#    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, \
#                                                    test_size = 0.2, random_state = 3)
    drop_list=['Survived','Name','Sex','Ticket','Cabin','PassengerId','Embarked']
    drop_list2=['Name','Sex','Ticket','Cabin','PassengerId','Embarked']

    X_train=df_train.drop(drop_list,axis=1)
    X_test=df_test.drop(drop_list2,axis=1)
    y_train=df_train['Survived']
    y_test=df_submission['Survived']

    # Normalize feature data
#    scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
#    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
#    X_test_scaled=scaler.transform(X_test)

    # One hot encode target values
    one_hot = OneHotEncoder()

    y_train_hot=y_train
    y_test_hot=y_test
#    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
#    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()


    #PS2 Classifier
    print('PS2 Accuracy')
    start_time=time.time()
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 3000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = False, clip_max = 5, max_attempts = 100, \
                                 random_state = None,restarts=10)#3)

    nn_model1.fit(X_train_scaled, y_train_hot)
    end_time=time.time()
    time_SA=end_time-start_time    
    print('TRAIN TIME, SA:', time_SA,end_time,start_time)
#    nn_model1.fit(X_train_scaled, y_train)

    from sklearn.metrics import accuracy_score

    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
#    y_train_accuracy = accuracy_score(y_train, y_train)

    print('training accuracy',y_train_accuracy)
    #0.45

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
#    y_test_accuracy = accuracy_score(y_test, y_test)

    print('testing accuracy',y_test_accuracy)

    print(confusion_matrix(y_test_hot,y_test_pred))
    print(classification_report(y_test_hot,y_test_pred))
#    print(pd.crosstab(y_test_hot,y_test_pred,rownames=['True'],colnames=['Predicted'],margins=True))    


    time_to_process=[time_SA+0.1,4.1]
    algorithm_labels=['SA','Backprop']
    plt.bar(algorithm_labels,time_to_process,width=0.8,bottom=None,align='center')
    plt.title('Time per 3000 Iterations, Neural Network Training')
#    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Algorithm')
    plt.ylabel('Time')
#    plt.ylim((0,10.0))
    plt.savefig('charts/NN_SA_iterationTime.png')
    plt.show()


    return


def neural_net_GA(df_sub,df_train,df_test):
    print("------ NEURAL NET, Genetic Algorithm ------")
    from sklearn.datasets import load_iris

    # Load the Iris dataset
#    data = load_iris()
    df_submission=df_sub
    df_train=df_train
    df_test=df_test

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

    # Split data into training and test sets
#    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, \
#                                                    test_size = 0.2, random_state = 3)
    drop_list=['Survived','Name','Sex','Ticket','Cabin','PassengerId','Embarked']
    drop_list2=['Name','Sex','Ticket','Cabin','PassengerId','Embarked']

    X_train=df_train.drop(drop_list,axis=1)
    X_test=df_test.drop(drop_list2,axis=1)
    y_train=df_train['Survived']
    y_test=df_submission['Survived']

    # Normalize feature data
#    scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
#    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
#    X_test_scaled=scaler.transform(X_test)

    # One hot encode target values
    one_hot = OneHotEncoder()

    y_train_hot=y_train
    y_test_hot=y_test
#    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
#    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()


    #PS2 Classifier
    print('PS2 Accuracy')
    from sklearn.metrics import accuracy_score


    #Gridsearch GA
    '''
    learning_rate_list=[0.01,0.1,0.25] #pop size originally at 500
    iters_list=[200,500,1000] #iters originally at 50 or something idk
    grid_results_train=np.zeros((len(learning_rate_list),len(iters_list)))
    grid_results_test=np.zeros((len(learning_rate_list),len(iters_list)))
    grid_results_time=np.zeros((len(learning_rate_list),len(iters_list)))
    i=0
    while i < len(learning_rate_list):
        lrate=learning_rate_list[i]
        print('learning rate:',lrate)
        j=0
        while j < len (iters_list):
            iters=iters_list[j]
            print('iters:',iters)
            start_time=time.time()                      #started at max_iters=3000, learning_rate=0.1
            nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 3000, \
                                 bias = True, is_classifier = True, learning_rate = lrate, \
                                 early_stopping = False, clip_max = 5, max_attempts = 100, \
                                 random_state = None,restarts=10,pop_size=iters)#3)
            end_time=time.time()
            nn_model1.fit(X_train_scaled, y_train_hot)
            # Predict labels for train set and assess accuracy
            y_train_pred = nn_model1.predict(X_train_scaled)

            y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

            print('training accuracy',y_train_accuracy)

            # Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test_scaled)

            y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

            print('testing accuracy',y_test_accuracy)
            print('time elapsed',end_time-start_time)
            grid_results_train[i,j]=y_train_accuracy
            grid_results_test[i,j]=y_test_accuracy
            grid_results_time[i,j]=end_time-start_time
            j=j+1
        i=i+1
    print('NN_GA grid results TRAIN:',grid_results_train)
    print('NN_GA best result:',np.amax(grid_results_train))
    print('NN_GA grid results TEST:',grid_results_test)
    '''

    start_time=time.time()                      #started at max_iters=3000, learning_rate=0.1
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 3000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = False, clip_max = 5, max_attempts = 100, \
                                 random_state = None,restarts=10,pop_size=200)#3)
    nn_model1.fit(X_train_scaled, y_train_hot)
    time_GA=end_time-start_time
    end_time=time.time()
    print('TRAIN TIME, GA:', time_GA,end_time,start_time)
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print('training accuracy',y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print('testing accuracy',y_test_accuracy)
    print('time elapsed',end_time-start_time)


    print(confusion_matrix(y_test_hot,y_test_pred))
    print(classification_report(y_test_hot,y_test_pred))
#    print(pd.crosstab(y_test_hot,y_test_pred,rownames=['True'],colnames=['Predicted'],margins=True))    

    time_to_process=[time_GA,4.1]
    algorithm_labels=['GA','Backprop']
    plt.bar(algorithm_labels,time_to_process,width=0.8,bottom=None,align='center')
    plt.title('Time per 3000 Iterations, Neural Network Training')
#    plt.legend(['MIMIC','RHC','SA','GA'])
    plt.xlabel('Algorithm')
    plt.ylabel('Time')
    #plt.ylim((0,1.0))
    plt.savefig('charts/NN_GA_iterationTime.png')
    plt.show()


    return




if __name__ == "__main__": 			  		 			 	 	 		 		 	  		   	  			  	
    main()

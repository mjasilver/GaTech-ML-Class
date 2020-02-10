
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


def main():

    df_submission,df_train,df_test=process_data() 
    neural_net(df_submission,df_train,df_test) #Done Setup
    decision_tree(df_submission,df_train,df_test) #Done Setup
    adaboost(df_submission,df_train,df_test) #Done Setup
    knn(df_submission,df_train,df_test) #Done Setup
    svm(df_submission,df_train,df_test) #Done Setup
    

def process_data():
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
    

def neural_net(df_sub,df_train,df_test):


    '''
    wine = pd.read_csv('wine_data.csv',names=["Cultivator","Alcohol","Malic_Acid","Ash","Alcalinity_of_Ash","Magnesium","Total_Phenols","Falvanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue","OD280","Proline"])

    #Look at the data
    wine.head()

    wine.describe().transpose()
    X = wine.drop('Cultivator',axis=1)
    y=wine['Cultivator']
    '''

    df_submission=df_sub
    df_train=df_train
    df_test=df_test

    drop_list=['Survived','Name','Sex','Ticket','Cabin','PassengerId','Embarked']
    drop_list2=['Name','Sex','Ticket','Cabin','PassengerId','Embarked']

    X_train=df_train.drop(drop_list,axis=1)
    X_test=df_test.drop(drop_list2,axis=1)
    y_train=df_train['Survived']
    y_test=df_submission['Survived']

    print(X_train.head())
    print(X_test.head())

    '''
    #Load Data
    iris = datasets.load_iris()
    X=iris.data
    y=iris.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=10)
    '''
    
    scaler = StandardScaler()

    #Fit to the training data
    scaler.fit(X_train)

    #Apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #GRIDSEARCH
    '''
    parameters = { 'hidden_layer_sizes':[(8,8,8),(13,13,13),(17,17,17)],
                   'activation':['relu'],
                   'max_iter':[3000],
                   'learning_rate_init':[0.0001,0.0005,0.001],
                   'solver':['adam']
        }

    mlp=MLPClassifier()
    clf=GridSearchCV(mlp,parameters)
    clf.fit(X_train,y_train)
    sorted(clf.cv_results_.keys())
    print(clf.cv_results_.keys())
    print('best estimator',clf.best_estimator_)
    print('best score',clf.best_score_)
    raise NotImplementedError
    '''       



    mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', max_iter=3000,learning_rate_init=0.001, solver='adam')
    mlp.fit(X_train,y_train)

    predictions = mlp.predict(X_test)

    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))

    print(pd.crosstab(y_test,predictions,rownames=['True'],colnames=['Predicted'],margins=True))    

    len(mlp.coefs_)
    len(mlp.coefs_[0])
    len(mlp.intercepts_[0])


    #PREP DATA FOR LOOPING
    df_X_train=pd.DataFrame(X_train)
    df_X_test=pd.DataFrame(X_test)
    df_y_train=pd.DataFrame(y_train)
    df_y_test=pd.DataFrame(y_test)
    X=df_X_train.append(df_X_test)
    y=df_y_train.append(df_y_test)

    #LEARNING CURVE: LOOP FOR DIFFERENT TRAINING SIZES    
    n_range=[0.01,0.02,0.1,0.25,0.4,0.5,0.6,0.75,0.9,0.98,0.99]#[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    scores = {}
    scores_list=[]
    train_scores_list=[]
    for n_size in n_range:
        print('n_range',n_size)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=n_size, random_state=10)#,random_state=4)
        mlp.fit(X_train,y_train)
        y_pred=mlp.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
        #Train scores, for learning curves
        y_pred_train=mlp.predict(X_train)
        train_scores_list.append(metrics.accuracy_score(y_train,y_pred_train))

    print("TRAINING SIZE")
    print('scores_lis',scores_list)
    a=['0.01','0.02','0.1','0.25','0.4','0.5','0.6','0.75','0.9','0.98','0.99']
    plt.plot(a,scores_list,a,train_scores_list)#(n_range,scores_list,n_range,train_scores_list)#n_range,scores_list)#,n_range,[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])#(['0.05','0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5'],scores_list)
    plt.title('Test Accuracy v. Train Accuracy, Neural Networks')
    plt.legend(['Test','Train'])
    plt.xlabel('Test Split')
    plt.ylabel('Accuracy')
    plt.ylim((0,1.0))
    plt.savefig('Titanic/mlp_Titanic_testSize.png')
    plt.show()
    


    #LEARNING CURVE: LOOP FOR DIFFERENT LEARNING RATES
    l_range=[0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.15]
    #scores = {}
    learning_list=[]
    for l_rate in l_range:
        print('l_rate',l_rate)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=10)#,random_state=4)
        mlp = MLPClassifier(hidden_layer_sizes=(13,13,13), activation='relu', max_iter=500,learning_rate_init=l_rate, solver='adam')
        mlp.fit(X_train,y_train)
        y_pred=mlp.predict(X_test)
        learning_list.append(metrics.accuracy_score(y_test,y_pred))

    print('LEARNING RATE')
    print('scores_lis',learning_list)
    plt.plot(l_range,learning_list)
    plt.xlabel('Learning Rate')
    plt.ylabel('Testing Accuracy')
    plt.ylim((0,1.0))
    plt.savefig('Titanic/mlp_Titanic_learningRate.png')
    plt.show()

    

    '''
    #LEARNING CURVE: LOOP FOR ITERATION TIME
    l_range=[0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.15,.25,.5]
    scores = {}
    scores_list=[]
    time_list=[]
    for l_rate in l_range:
        print('I_TIME',l_rate)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=10)#,random_state=4)
        mlp = MLPClassifier(hidden_layer_sizes=(13,13,13), activation='relu', max_iter=10000,learning_rate_init=l_rate, solver='adam')
        start_time=time.time()
        mlp.fit(X_train,y_train)
        end_time=time.time()
        elapsed=end_time-start_time
        y_pred=mlp.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
        time_list.append(elapsed)
#    print('elapsed',end_time-start_time)
        
    print('scores_list',scores_list)
    print('time_list',time_list)
    plt.plot(l_range,time_list)
    plt.xlabel('Learning Rate')
    plt.ylabel('Time Elapsed')
    #plt.ylim((0,1.0))
    plt.savefig('Titanic/mlp_Titanic_timeElapsed.png')
    #plt.show()
    '''



    #LEARNING CURVE: LOOP FOR DIFFERENT network architectures
    l_range=[(4,4,4),(8,8,8),(13,13,13),(30,30,30)]
    #scores = {}
    learning_list=[]
    time_list=[]
    for l_rate in l_range:
        print('l_rate',l_rate)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=10)#,random_state=4)
        mlp = MLPClassifier(hidden_layer_sizes=l_rate, activation='relu', max_iter=3000,learning_rate_init=0.0001, solver='adam')
        start_time=time.time()
        mlp.fit(X_train,y_train)
        end_time=time.time()
        elapsed=end_time-start_time
        time_list.append(elapsed)
        y_pred=mlp.predict(X_test)
        learning_list.append(metrics.accuracy_score(y_test,y_pred))

    print('LEARNING RATE')
    print('scores_lis',learning_list)
    plt.plot(['(4,4,4)','(8,8,8)','(13,13,13)','(30,30,30)'],learning_list)#l_range,learning_list)
    plt.title('Neural Network Accuracy at Varying Architectures (Titanic)')
    plt.xlabel('Hidden Layer Architecture')
    plt.ylabel('Testing Accuracy')
    plt.ylim((0,1.0))
    plt.savefig('Titanic/mlp_Titanic_networkArchitecture.png')
    plt.show()

    plt.plot(['(4,4,4)','(8,8,8)','(13,13,13)','(30,30,30)'],time_list)
    plt.xlabel('Hidden Layer Architecture')
    plt.ylabel('Training Time (sec)')
    plt.title('Neural Network Training Time at Varying Architectures (Titanic)')
#    plt.ylim((0,0.01))
    plt.savefig('Titanic/mlp_Titanic_trainingTime.png')
    plt.show()


#    raise NotImplementedError
    return 

    

def decision_tree(df_sub,df_train,df_test):


    '''
    wine = pd.read_csv('wine_data.csv',names=["Cultivator","Alcohol","Malic_Acid","Ash","Alcalinity_of_Ash","Magnesium","Total_Phenols","Falvanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue","OD280","Proline"])

    #Look at the data
    wine.head()

    wine.describe().transpose()
    X = wine.drop('Cultivator',axis=1)
    y=wine['Cultivator']
    '''

    df_submission=df_sub
    df_train=df_train
    df_test=df_test

    drop_list=['Survived','Name','Sex','Ticket','Cabin','PassengerId','Embarked']
    drop_list2=['Name','Sex','Ticket','Cabin','PassengerId','Embarked']

    X_train=df_train.drop(drop_list,axis=1)
    X_test=df_test.drop(drop_list2,axis=1)
    y_train=df_train['Survived']
    y_test=df_submission['Survived']

    print(X_train.head())
    print(X_test.head())


    '''
    col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
    pima = pd.read_csv("pima-indians-diabetes.csv",header=None,names=col_names)
    pima.head()

    #split dataset in features and target variable
    feature_cols=['pregnant','insulin','bmi','age','glucose','bp','pedigree']
    
    X = pima[feature_cols]
    y=pima.label
    '''

    #Create DEcision Tree classifier object
    clf = DecisionTreeClassifier(max_depth=5,min_samples_leaf=10)

    #Train Decision Tree Classifier
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)

    print('Training Accuracy: ',metrics.accuracy_score(y_train,y_pred))

    y_pred = clf.predict(X_test)

    #Model Accuracy, how often is the classifier correct?
    print("Test Accuracy:",metrics.accuracy_score(y_test,y_pred))

    print(pd.crosstab(y_test,y_pred,rownames=['True'],colnames=['Predicted'],margins=True))    


    #PREP DATA FOR LOOPING
    df_X_train=pd.DataFrame(X_train)
    df_X_test=pd.DataFrame(X_test)
    df_y_train=pd.DataFrame(y_train)
    df_y_test=pd.DataFrame(y_test)
    X=df_X_train.append(df_X_test)
    y=df_y_train.append(df_y_test)


    #LEARNING CURVE: LOOP FOR DIFFERENT TRAINING SIZES
#    n_range=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    n_range=[0.01,0.02,0.1,0.25,0.4,0.5,0.6,0.75,0.9,0.98,0.99]#[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    scores = {}
    scores_list=[]
    train_scores_list=[]
    for n_size in n_range:
        print('n_range',n_size)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=n_size, random_state=10)#,random_state=4)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
        #Train scores, for learning curves
        y_pred_train=clf.predict(X_train)
        train_scores_list.append(metrics.accuracy_score(y_train,y_pred_train))

    print("TRAINING SIZE")
    print('scores_lis',scores_list)
    a=['0.01','0.02','0.1','0.25','0.4','0.5','0.6','0.75','0.9','0.98','0.99']
    plt.plot(a,scores_list,a,train_scores_list)#(n_range,scores_list,n_range,train_scores_list)#n_range,scores_list)#,n_range,[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])#(['0.05','0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5'],scores_list)
#    plt.plot(n_range,scores_list,n_range,train_scores_list)
    plt.title('Test Accuracy v. Train Accuracy, Decision Trees (Titanic)')
    plt.legend(['Test','Train'])
    plt.xlabel('Test Split')
    plt.ylabel('Accuracy')
    plt.ylim((0,1.0))
    plt.savefig('Titanic/mlp_Titanic_testSize.png')
    plt.show()



    #LEARNING CURVE: LOOP FOR DIFFERENT Max Depths
    l_range=[1,5,10,15,20]
    #scores = {}
    learning_list=[]
    time_list=[]
    for l_rate in l_range:
        print('l_rate',l_rate)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=10)#,random_state=4)
        clf = DecisionTreeClassifier(max_depth=l_rate, min_samples_leaf=10)
        start_time=time.time()
        clf.fit(X_train,y_train)
        end_time=time.time()
        elapsed=end_time-start_time
        time_list.append(elapsed)
        y_pred=clf.predict(X_test)
        learning_list.append(metrics.accuracy_score(y_test,y_pred))

    print('LEARNING RATE')
    print('scores_lis',learning_list)
    plt.plot(['1','5','10','15','20'],learning_list)#l_range,learning_list)
    plt.title('Decision Tree Accuracy at Varying Max Depths (Titanic)')
    plt.xlabel('Max Depth')
    plt.ylabel('Testing Accuracy')
    plt.ylim((0,1.0))
    plt.savefig('Titanic/dt_Titanic_learningRate.png')
    plt.show()

    plt.plot(['1','5','10','15','20'],time_list)
    plt.xlabel('Max Depth')
    plt.ylabel('Training Time (sec)')
    plt.title('Decision Tree Training Time at Varying Max Depths (Titanic)')
#    plt.ylim((0,0.01))
    plt.savefig('Titanic/dt_Titanic_trainingTime.png')
    plt.show()




    import graphviz
    '''
    dot_data = tree.export_graphviz(clf,out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    '''
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    import pydotplus
    
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file = dot_data, filled=True, rounded=True,
                    special_characters=True, #feature_names = feature_cols,
                    #class_names=['0','1'])
                         )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('Titanic/decisionTree_Titanic.png')
    Image(graph.create_png())
    
    
    print('after graphviz')

    #Section: Optimizing Decision Tree Performance
    '''
    #Create Decision Tree Classifier Object
    clf = DecisionTreeClassifier(criterion="entropy",max_depth=5)  #min_samples_leaf can be set to 5%, max_leaf nodes can also be set
    clf=clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    #Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
    '''


    '''
    dot_data = StringIO()
    export_graphviz(clf, out_file = dot_data, filled=True, rounded=True,
                    special_characters=True, feature_names = feature_cols,
                    class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('diabetes.png')
    Image(graph.create_png())
    '''
          
#    raise NotImplementedError
    return

def adaboost(df_sub,df_train,df_test):


    '''
    wine = pd.read_csv('wine_data.csv',names=["Cultivator","Alcohol","Malic_Acid","Ash","Alcalinity_of_Ash","Magnesium","Total_Phenols","Falvanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue","OD280","Proline"])

    #Look at the data
    wine.head()

    wine.describe().transpose()
    X = wine.drop('Cultivator',axis=1)
    y=wine['Cultivator']
    '''

    df_submission=df_sub
    df_train=df_train
    df_test=df_test

    drop_list=['Survived','Name','Sex','Ticket','Cabin','PassengerId','Embarked']
    drop_list2=['Name','Sex','Ticket','Cabin','PassengerId','Embarked']

    X_train=df_train.drop(drop_list,axis=1)
    X_test=df_test.drop(drop_list2,axis=1)
    y_train=df_train['Survived']
    y_test=df_submission['Survived']

    print(X_train.head())
    print(X_test.head())

    #Create adaboost classifier object
    abc=AdaBoostClassifier(n_estimators=50,learning_rate=1)

    #Train Adaboost Classifier
    model=abc.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = model.predict(X_test)

    #Model Accuracy, how often is the classifer correct?
    print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
    print(pd.crosstab(y_test,y_pred,rownames=['True'],colnames=['Predicted'],margins=True))    
#    raise NotImplementedError

    #PREP DATA FOR LOOPING
    df_X_train=pd.DataFrame(X_train)
    df_X_test=pd.DataFrame(X_test)
    df_y_train=pd.DataFrame(y_train)
    df_y_test=pd.DataFrame(y_test)
    X=df_X_train.append(df_X_test)
    y=df_y_train.append(df_y_test)

    #LEARNING CURVE: LOOP FOR DIFFERENT TRAINING SIZES
    n_range=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    scores = {}
    scores_list=[]
    train_scores_list=[]
    for n_size in n_range:
        print('n_range',n_size)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=n_size, random_state=10)#,random_state=4)
        abc.fit(X_train,y_train)
        y_pred=abc.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
        #Train scores, for learning curves
        y_pred_train=abc.predict(X_train)
        train_scores_list.append(metrics.accuracy_score(y_train,y_pred_train))

    print("TRAINING SIZE")
    print('scores_lis',scores_list)
    plt.plot(n_range,scores_list,n_range,train_scores_list)
    plt.title('Test Accuracy v. Train Accuracy, Adaboost')
    plt.legend(['Test','Train'])
    plt.xlabel('Test Split')
    plt.ylabel('Accuracy')
    plt.ylim((0,1.0))
    plt.savefig('Titanic/abc_Titanic_testSize.png')
    plt.show()



    #LEARNING CURVE: LOOP FOR DIFFERENT n_estimators
    l_range=[1,5,10,25,50,100,250]
    #scores = {}
    learning_list=[]
    time_list=[]
    for l_rate in l_range:
        print('l_rate',l_rate)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=10)#,random_state=4)
        abc=AdaBoostClassifier(n_estimators=l_rate,learning_rate=1)
        start_time=time.time()
        abc.fit(X_train,y_train)
        end_time=time.time()
        elapsed=end_time-start_time
        time_list.append(elapsed)
        y_pred=abc.predict(X_test)
        learning_list.append(metrics.accuracy_score(y_test,y_pred))

    print('LEARNING RATE')
    print('scores_lis',learning_list)
    plt.plot(['1','5','10','25','50','100','250'],learning_list)#l_range,learning_list)
    plt.title('Adaboost Accuracy at Varying Number Estimators (Titanic)')
    plt.xlabel('Number Estimators')
    plt.ylabel('Testing Accuracy')
    plt.ylim((0,1.0))
    plt.savefig('Titanic/abc_Titanic_learningRate.png')
    plt.show()

    plt.plot(['1','5','10','25','50','100','250'],time_list)
    plt.xlabel('Number Estimators')
    plt.ylabel('Training Time (sec)')
    plt.title('Adaboost Training Time at Varying Number Estimators (Titanic)')
#    plt.ylim((0,0.01))
    plt.savefig('Titanic/abc_Titanic_trainingTime.png')
    plt.show()




    #Using Different Base Learners
    from sklearn.svm import SVC

    svc=SVC(probability=True,kernel='linear')

    #Create adaboost classifier object
    abc = AdaBoostClassifier(n_estimators=50,base_estimator=svc,learning_rate=1)

    #Train Adaboost Classifier
    model = abc.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred=model.predict(X_test)
        
    print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
#    raise NotImplementedError
    return

def knn(df_sub,df_train,df_test):


    '''
    wine = pd.read_csv('wine_data.csv',names=["Cultivator","Alcohol","Malic_Acid","Ash","Alcalinity_of_Ash","Magnesium","Total_Phenols","Falvanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue","OD280","Proline"])

    #Look at the data
    wine.head()

    wine.describe().transpose()
    X = wine.drop('Cultivator',axis=1)
    y=wine['Cultivator']
    '''

    df_submission=df_sub
    df_train=df_train
    df_test=df_test

    drop_list=['Survived','Name','Sex','Ticket','Cabin','PassengerId','Embarked']
    drop_list2=['Name','Sex','Ticket','Cabin','PassengerId','Embarked']

    X_train=df_train.drop(drop_list,axis=1)
    X_test=df_test.drop(drop_list2,axis=1)
    y_train=df_train['Survived']
    y_test=df_submission['Survived']

    print(X_train.head())
    print(X_test.head())

    k_range = range(1,25)
    scores = {}
    scores_list = []
    time_list=[]
    train_scores_list=[]
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
        start_time=time.time()
        knn.fit(X_train,y_train)
        end_time=time.time()
        print('k ',k,'start',start_time,'end',end_time)
        y_pred=knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        elapsed=end_time-start_time
        print('k = ',k,'elapsed = ',elapsed)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
        time_list.append(elapsed)
        print(pd.crosstab(y_test,y_pred,rownames=['True'],colnames=['Predicted'],margins=True))    
        #Train scores, for learning curves
        y_pred_train=knn.predict(X_train)
        train_scores_list.append(metrics.accuracy_score(y_train,y_pred_train))

    print('scores_list',scores_list)
    print('time_list',time_list)
#   %matplotlib inline
    import matplotlib.pyplot as plt

    plt.plot(k_range,scores_list,k_range,train_scores_list)
    plt.title('Test Accuracy v. Train Accuracy, KNN')
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Accuracy')
    plt.legend(['Test','Train'])
    plt.ylim((0,1.0))
    plt.savefig('Titanic/knn_Titanic_testSize.png')
    plt.show()

    plt.plot(k_range,time_list)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Training Time')
    plt.title('KNN Training Time at Varying N-Neighbors (Titanic)')
    #plt.ylim((0,1.0))
    plt.savefig('Titanic/knn_Titanic_trainingTime.png')
    plt.show()



    #PREP DATA FOR LOOPING
    df_X_train=pd.DataFrame(X_train)
    df_X_test=pd.DataFrame(X_test)
    df_y_train=pd.DataFrame(y_train)
    df_y_test=pd.DataFrame(y_test)
    X=df_X_train.append(df_X_test)
    y=df_y_train.append(df_y_test)

    #LEARNING CURVE: LOOP FOR DIFFERENT TRAINING SIZES
    n_range=[0.01,0.02,0.1,0.25,0.4,0.5,0.6,0.75,0.9,0.98,0.99]#[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    scores = {}
    scores_list=[]
    time_list=[]
    train_scores_list=[]
    for n_size in n_range:
        knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=n_size, random_state=10)#,random_state=4)
        start_time=time.time()
        knn.fit(X_train,y_train)
        end_time=time.time()
        elapsed=end_time-start_time
        time_list.append(elapsed)
        y_pred=knn.predict(X_test)
        #scores[k]=metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
        #Train scores, for learning curves
        y_pred_train=knn.predict(X_train)
        train_scores_list.append(metrics.accuracy_score(y_train,y_pred_train))

    print('scores_list',scores_list)
    a=['0.01','0.02','0.1','0.25','0.4','0.5','0.6','0.75','0.9','0.98','0.99']
    plt.plot(a,scores_list,a,train_scores_list)#(n_range,scores_list,n_range,train_scores_list)#n_range,scores_list)#,n_range,[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])#(['0.05','0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5'],scores_list)
    plt.xlabel('Test Split')
    plt.ylabel('Testing Accuracy')
    plt.ylim((0,1.0))
    plt.savefig('Titanic/knn_Titanic_testSize.png')
    plt.title('KNN Testing Accuracy at Varying Test Sizes')
    plt.show()
        
    plt.plot(n_range,time_list)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Training Time')
    plt.title('KNN Training Time at Varying Training Sizes (Titanic)')
#    plt.ylim((0,0.01))
    plt.savefig('Titanic/knn_Titanic_trainingTime.png')
    plt.show()



    '''
    knn=KNeighborsClassifier(n_neighbors=5)
    knn.fit(X,y)

    classes={0:'setosa',1:'versicolor',2:'viginica'}
    x_new=[[3,4,5,2],[5,4,2,2]]
    y_predict=knn.predict(x_new)
    print('predictions are:')
    print(classes[y_predict[0]])
    print(classes[y_predict[1]])
    '''
#    raise NotImplementedError
    return

def svm(df_sub,df_train,df_test):


    '''
    wine = pd.read_csv('wine_data.csv',names=["Cultivator","Alcohol","Malic_Acid","Ash","Alcalinity_of_Ash","Magnesium","Total_Phenols","Falvanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue","OD280","Proline"])

    #Look at the data
    wine.head()

    wine.describe().transpose()
    X = wine.drop('Cultivator',axis=1)
    y=wine['Cultivator']
    '''

    df_submission=df_sub
    df_train=df_train
    df_test=df_test

    drop_list=['Survived','Name','Sex','Ticket','Cabin','PassengerId','Embarked']
    drop_list2=['Name','Sex','Ticket','Cabin','PassengerId','Embarked']

    X_train=df_train.drop(drop_list,axis=1)
    X_test=df_test.drop(drop_list2,axis=1)
    y_train=df_train['Survived']
    y_test=df_submission['Survived']

    print(X_train.head())
    print(X_test.head())


    #Create an SVM Classifier
    clf = sklearn.svm.SVC(kernel='linear', C=1, gamma=1)

    #Train the model using the training sets
    clf.fit(X_train,y_train)

    #Predict the response for test datset
    y_pred = clf.predict(X_test)

    #Model Accuracy: how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

    #Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test,y_pred))
    #Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test,y_pred))
    print(pd.crosstab(y_test,y_pred,rownames=['True'],colnames=['Predicted'],margins=True))    


    #PREP DATA FOR LOOPING
    df_X_train=pd.DataFrame(X_train)
    df_X_test=pd.DataFrame(X_test)
    df_y_train=pd.DataFrame(y_train)
    df_y_test=pd.DataFrame(y_test)
    X=df_X_train.append(df_X_test)
    y=df_y_train.append(df_y_test)

    #LEARNING CURVE: LOOP FOR DIFFERENT TRAINING SIZES
    n_range=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    scores = {}
    scores_list=[]
    train_scores_list=[]
    for n_size in n_range:
        print('looping at',n_size)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=n_size, random_state=10)#,random_state=4)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
        #Train scores, for learning curves
        y_pred_train=clf.predict(X_train)
        train_scores_list.append(metrics.accuracy_score(y_train,y_pred_train))
        #print(pd.crosstab(y_test,y_pred,rownames=['True'],colnames=['Predicted'],margins=True))    

    print("TRAINING SIZE")
    print('scores_lis',scores_list)
    plt.plot(n_range,scores_list,n_range,train_scores_list)
    plt.title('Test Accuracy v. Train Accuracy, SVMs')
    plt.legend(['Test','Train'])
    plt.xlabel('Test Split')
    plt.ylabel('Accuracy')
    plt.ylim((0,1.0))
    plt.savefig('Titanic/svm_Titanic_testSize.png')
    plt.show()

    
    return 



if __name__ == "__main__": 			  		 			 	 	 		 		 	  		   	  			  	
    main()

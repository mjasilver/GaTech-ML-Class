
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
import scipy
from scipy import stats
from sklearn.feature_selection import VarianceThreshold

def main():
#Neural Net
    df_X,df_y=import_data()
#    df_X,df_y=process_data2()
#    print('Beginning NN')    
#    neural_net(df_X,df_y)
#    raise NotImplementedError

#    print('df_X columns:',df_X.columns)
    col_names=df_X.columns
#    print('df_X means:',df_X.mean())
    col_means=df_X.mean()
#    print('column names',col_names)
#    print('column means',col_means)
    col_std=df_X.std()

    print('column names',col_names)
    print('column means',col_means)
    print('column stdev',col_std)
#    raise NotImplementedError

    
    #Part 1            
    k_means_labels=k_means(df_X,df_y)
    exp_max_labels=exp_max(df_X,df_y)
    
#    raise NotImplementedError

    
    #Part 2
#    print(df_X.head())
    
    x_PCA=principle_components_analysis(df_X)
    x_ICA=independent_components_analysis(df_X)    
    x_RP=random_projection(df_X)
    x_VT=variance_threshold(df_X)
    

    
    #Part 3
        
#    print('Running KMEANS on dimensionality reductions...')
#    print('PCA')
#    print('df_y head',df_y.head())
    k_means_PCA=k_means(x_PCA,df_y)
#    print('ICA')
    k_means_ICA=k_means(x_ICA,df_y)
#    print('RP')
    
    k_means_RP=k_means(x_RP,df_y)
    
    kmeans_VT=k_means(x_VT,df_y)
    
#    print('column names',col_names)
#    print('column means',col_means)
    

        
    print('Running EXPECATION MAXIMIZATION on dimensionality reductions...')
    print('PCA')
    em_PCA=exp_max(x_PCA,df_y)
#    exp_max(x_PCA)
#    print('ICA')
    em_ICA=exp_max(x_ICA,df_y)
#    print('RP')
    em_RP=exp_max(x_RP,df_y)
    em_VT=exp_max(x_VT,df_y)
    

    #PART 4
    
#    print('------PCA on NN------')
    neural_net(x_PCA,df_y)
#    print('ICA on NN')
    neural_net(x_ICA,df_y)
#    print('RP on NN')
    neural_net(x_RP,df_y)
    
    
    #PART 5
    
    print('Beginning NN')    

    print('------PCA on NN------')
    
#    neural_net_RHC(df_X,df_y)
#    print('type kmeans_pca:',type(k_means_PCA))
#    print('type x_PCA',type(x_PCA))
    x_PCA=pd.DataFrame(x_PCA)
#    print('type x_PCA',type(x_PCA))
#    print('x_PCA:',x_PCA)
    x_PCA['clustering']=k_means_PCA
    neural_net(x_PCA,df_y)#(x_PCA,df_y)
    
    x_PCA=pd.DataFrame(x_PCA)
    x_PCA['clustering']=em_PCA
    neural_net(x_PCA,df_y)#(x_PCA,df_y)
    
    


#    print('------ICA on NN------')   
    
    x_ICA=pd.DataFrame(x_ICA)
    x_ICA['clustering']=k_means_ICA
    neural_net(x_ICA,df_y)#(x_PCA,df_y)

    x_ICA=pd.DataFrame(x_ICA)
    x_ICA['clustering']=em_ICA
    neural_net(x_ICA,df_y)#(x_PCA,df_y)
    


#    print('------RP on NN------')
    
    x_RP=pd.DataFrame(x_RP)
    x_RP['clustering']=k_means_RP
    neural_net(x_RP,df_y)#(x_PCA,df_y)
    x_RP=pd.DataFrame(x_RP)
    x_RP['clustering']=em_RP
    neural_net(x_RP,df_y)#(x_PCA,df_y)
    
    
#    print('df_y type:',type(df_y))
#    print('df_y head:',df_y.head())
#    neural_net(df_X,df_y)#(x_PCA,df_y)
    

def import_data():
#    print(os.getcwd())
#    print(os.listdir("Titanic"))

#    df_results=pd.read_csv('horse_race_data/results.csv')
    df_submission=pd.read_csv('UFC_Data/randomized_corner.csv')

    df_y=df_submission['Winner']
    df_y=df_y.to_frame()
    drop_list=['Winner']
    df_X=df_submission.drop(drop_list,axis=1)

#    print('df_y Head')
#    print(df_y.head())
#    print('df_X Head')
#    print(df_X.head())
#    print('type y',type(df_y))
#    print('type X',type(df_X))
    print('done importing data')
    return df_X,df_y


def process_data2():
#    print(os.getcwd())
#    print(os.listdir("Titanic"))

#    df_results=pd.read_csv('horse_race_data/results.csv')
#    df_submission=pd.read_csv('Titanic/gender_submission.csv')
#    df_test=pd.read_csv('Titanic/test.csv')
#    df_train=pd.read_csv('Titanic/train.csv')
    df_combined=pd.read_csv('Titanic/Combined.csv')
    

    #DEAL WITH N/A VALUES
#    print(df_test.query("PassengerId=='902'"))
    mean_age=df_combined['Age'].mean(axis=0,skipna=True)
    mean_fare=df_combined['Fare'].mean(axis=0,skipna=True)
    df_combined['Age'].fillna(value=mean_age,inplace=True)
    df_combined['Fare'].fillna(value=mean_fare,inplace=True)


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
    unique_sex=df_combined['Sex'].unique()
    unique_embarked=df_combined['Embarked'].unique()
    unique_Pclass=df_combined['Pclass'].unique()
    #TRAIN SET
    #SEX
    df_combined['Is_Male']=0
    df_combined.loc[df_combined['Sex']=='male','Is_Male']=1
    #EMBARKED
    df_combined['Is_Embarked_S']=0
    df_combined.loc[df_combined['Embarked']=='S','Is_Embarked_S']=1
    df_combined['Is_Embarked_C']=0
    df_combined.loc[df_combined['Embarked']=='C','Is_Embarked_C']=1
    df_combined['Is_Embarked_Q']=0
    df_combined.loc[df_combined['Embarked']=='Q','Is_Embarked_Q']=1
    #PCLASS
    df_combined['Is_Pclass_1']=0
    df_combined['Is_Pclass_2']=0
    df_combined['Is_Pclass_3']=0
    df_combined.loc[df_combined['Pclass']==1,'Is_Pclass_1']=1
    df_combined.loc[df_combined['Pclass']==2,'Is_Pclass_2']=1
    df_combined.loc[df_combined['Pclass']==3,'Is_Pclass_3']=1
    
#    print(df_combined.head())

    drop_list=['Survived','Name','Sex','Ticket','Cabin','PassengerId','Embarked']
    drop_list2=['Name','Sex','Ticket','Cabin','PassengerId','Embarked']

    df_y=df_combined['Survived']
    df_X=df_combined.drop(drop_list,axis=1)
    df_y=df_y.to_frame()

#    print('df_y head',df_y.head())
#    print('df_X head',df_X.head())

#    enc = preprocessing.OneHotEncoder(categories=[unique_sex,unique_embarked,unique_Pclass])
#    print(df_train.head())

    #THEN, DATASET IS READY!!!!!
    
    return df_X, df_y


def k_means(data_set,y):
    #https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
    print("-----KMEANS------")
#    X=data_set
    min_max_scaler=preprocessing.MinMaxScaler()
    x_scaled=min_max_scaler.fit_transform(data_set.copy())
    X=pd.DataFrame(x_scaled)

    i_vector=[]
    wcss=[]
    silhouette_vector=[]
    number_components=20#len(X.columns)-1

    for i in range(2,number_components):
        kmeans=sklearn.cluster.KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
#        predictions=kmeans.fit_predict(X)
#        print('labels',kmeans.labels_)
        labels=kmeans.labels_
#        print('cluster center',kmeans.cluster_centers_)
        silhouette=sklearn.metrics.silhouette_score(X,labels)
        silhouette_vector.append(silhouette)
        i_vector.append(i)
    #Elbow Method
    plt.plot(i_vector,wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    #Silhouette Score
    plt.plot(i_vector,silhouette_vector)#(range(1,number_components),silhouette_vector)
    plt.title('Silhouette Score by Number of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()


#Clusters chosen and Information Gain from clusters
    clusters = 20
    kmeans=sklearn.cluster.KMeans(n_clusters=clusters,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    labels=kmeans.labels_
    df_comp=y.copy()
    df_comp['Cluster']=labels
    i_vector=[]
    survival_vector=[]
    infogain_vector=[]
    entropy_vector=[]
    entropy_base=[]
    survival_base=[]
    for i in range(0,clusters):
        con1=df_comp['Cluster']==i
        con2=df_comp['Winner']==1
    #    print('size',df_comp['Cluster'][con1].size)
    #    print('condition 1',df_comp[con1])
        survived=df_comp[con1 & con2].size/2
        total=df_comp[con1].size/2#.count
        entropy_before=-(1814/3592*np.log2(1814/3592)+1778/3592*np.log2(1778/3592))
        entropy_new=-(survived/total*np.log2(survived/total)+(total-survived)/total*np.log2((total-survived)/total))
        if survived==total or survived==0:
            entropy_new=0
        print('survived',survived)
        print('total',total)
        print('entropy before',entropy_before)
        print('entropy after',entropy_new)
        print('entropy test',-(0.5*np.log2(0.5)+0.5*np.log2(0.5)))
#        print('condition 1 & 2',df_comp[con1])#[con1 & con2])
#        print('label length',len(labels))
#        print('type',type(df_comp[con1 & con2]))
#        print(df_comp.head())
        i_vector.append(i)
        survival_vector.append(survived/total)
        entropy_vector.append(entropy_new)
        entropy_base.append(entropy_before)
        survival_base.append(1814/3592)
        i=i+1

    #Entropy from each cluster
    #Here, you're going to do the base entropy as a flat line across the graph, then chart the entropy for each cluster
    plt.plot(i_vector,entropy_vector,i_vector,entropy_base)#(range(1,number_components),silhouette_vector)
    plt.title('Entropy by Cluster')#'Information Gain by Cluster'
    plt.xlabel('Number of clusters')
    plt.ylabel('Entropy')
    plt.legend(['Clusters','Baseline'])
    plt.ylim((0,1.0))
    plt.show()
    
    #Survival from each cluster
    #Here, you're going to chart the base survival rate as a flat line across the chart, then plot the survival rate for each cluster
    plt.plot(i_vector,survival_vector,i_vector,survival_base)#(range(1,number_components),silhouette_vector)
    plt.title('Winning Percentage by Cluster')
    plt.xlabel('Number of clusters')
    plt.ylabel('Winning Percentage')
    plt.legend(['Clusters','Baseline'])
    plt.ylim((0,1.0))
    plt.show()
    
    best_cluster=np.argmin(entropy_vector)
    print('Best Cluster:',best_cluster)#np.argmin(entropy_vector))
    print('Means:',kmeans.cluster_centers_[best_cluster,:])

    return labels

def exp_max(data_set,y):
    #https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html
    print('------EXPECTATION MAXIMIZATION------')
    #X=data_set
    min_max_scaler=preprocessing.MinMaxScaler()
    x_scaled=min_max_scaler.fit_transform(data_set.copy())
    X=pd.DataFrame(x_scaled)
#    print(X.head())
#    clf=sklearn.mixture.GaussianMixture(n_components=5,covariance_type='full')
#    clf.fit(X)

#    labels=clf.predict(X)
#    print('labels',len(labels))
#    print('labels',labels)

    
    '''
    x = np.linspace(-20., 30.)
    y = np.linspace(-20., 40.)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X[:, 0], X[:, 1], .8)

    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()    
    '''


    i_vector=[]
    wcss=[]
    silhouette_vector=[]
    number_components=30#len(X.columns)-1

    for i in range(2,number_components):
        clf=sklearn.mixture.GaussianMixture(n_components=5,covariance_type='full')
        clf.fit(X)
#        wcss.append(kmeans.inertia_)
#        predictions=kmeans.fit_predict(X)
#        print('labels',kmeans.labels_)
        labels=clf.predict(X)
#        print('cluster center',kmeans.cluster_centers_)
        silhouette=sklearn.metrics.silhouette_score(X,labels)
        silhouette_vector.append(silhouette)
        i_vector.append(i)
    #Elbow Method
#    plt.plot(i_vector,wcss)
#    plt.title('Elbow Method')
#    plt.xlabel('Number of clusters')
#    plt.ylabel('WCSS')
#    plt.show()

    #Silhouette Score
    plt.plot(i_vector,silhouette_vector)#(range(1,number_components),silhouette_vector)
    plt.title('Silhouette Score by Groupings')
    plt.xlabel('Groupings')
    plt.ylabel('Silhouette Score')
    plt.show()


#Clusters chosen and Information Gain from clusters
    clusters = 20#8
    clf=sklearn.mixture.GaussianMixture(n_components=clusters,covariance_type='full')
    clf.fit(X)
    labels=clf.predict(X)
    print('labels',labels)
    df_comp=y.copy()
    df_comp['Cluster']=labels
    i_vector=[]
    survival_vector=[]
    infogain_vector=[]
    entropy_vector=[]
    entropy_base=[]
    survival_base=[]
    for i in range(0,clusters):
        print('cluster:',i)
        con1=df_comp['Cluster']==i
        con2=df_comp['Winner']==1
    #    print('size',df_comp['Cluster'][con1].size)
    #    print('condition 1',df_comp[con1])
        survived=df_comp[con1 & con2].size/2
        total=df_comp[con1].size/2#.count
        entropy_before=-(1778/3592*np.log2(1778/3592)+1814/3592*np.log2(1814/3592))
        entropy_new=-(survived/total*np.log2(survived/total)+(total-survived)/total*np.log2((total-survived)/total))
        if survived==total or survived==0:
            entropy_new=0
        print('survived',survived)
        print('total',total)
        print('entropy before',entropy_before)
        print('entropy after',entropy_new)
#        print('condition 1 & 2',df_comp[con1])#[con1 & con2])
#        print('label length',len(labels))
#        print('type',type(df_comp[con1 & con2]))
#        print(df_comp.head())
        i_vector.append(i)
        survival_vector.append(survived/total)
        entropy_vector.append(entropy_new)
        entropy_base.append(entropy_before)
        survival_base.append(1814/3592)
        i=i+1

    #Entropy from each cluster
    #Here, you're going to do the base entropy as a flat line across the graph, then chart the entropy for each cluster
    plt.plot(i_vector,entropy_vector,i_vector,entropy_base)#(range(1,number_components),silhouette_vector)
    plt.title('Entropy by Cluster')#'Information Gain by Cluster'
    plt.xlabel('Number of clusters')
    plt.ylabel('Entropy')
    plt.legend(['Clusters','Baseline'])
    plt.ylim((0,1.0))
    plt.show()
    
    #Survival from each cluster
    #Here, you're going to chart the base survival rate as a flat line across the chart, then plot the survival rate for each cluster
    plt.plot(i_vector,survival_vector,i_vector,survival_base)#(range(1,number_components),silhouette_vector)
    plt.title('Winning Percentage by Cluster')
    plt.xlabel('Number of clusters')
    plt.ylabel('Winning Percentage')
    plt.legend(['Clusters','Baseline'])
    plt.ylim((0,1.0))
    plt.show()

    best_cluster=np.argmin(entropy_vector)
    print('Best Cluster:',best_cluster)#np.argmin(entropy_vector))
    print('Means:',clf.means_[best_cluster,:])

    return labels

def principle_components_analysis(data_set):
    #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    print('------PCA------')
    X=data_set.copy()

    #Need to standardize features first???
    #https://www.youtube.com/watch?v=kApPBm1YsqU

    print('Columns',len(X.columns))

    variance_vector=[]
    i_vector=[]
    loss_vector=[]
    eigenvalues_vector=[]
    i=1
    while i<30:#len(X.columns):
        pca=PCA(n_components=i)
        X_transformed=pca.fit_transform(X)
        X_inverse_transformed=pca.inverse_transform(X_transformed)
#        print('X_trans i',X_transformed[0:5,:])
#        print('X_inv_trans i',X_transformed[0:5,:])
#        loss_calc=np.max((X_inverse_transformed-X))        
        loss_calc=((X_inverse_transformed-X)**2).sum()
        loss_calc=loss_calc.sum()
        var_val=sum(pca.explained_variance_ratio_)
        variance_vector.append(var_val)
        i_vector.append(i)
        loss_vector.append(loss_calc)
        eigenvalues_vector.append(pca.explained_variance_)
        print('variance',i,':',var_val)
        print('loss calc',loss_calc)#loss_calc.sum())
        print('eigenvalues',i,':',pca.explained_variance_)
        i=i+1

    #CHART: Explained Variance Percentage
    plt.plot(i_vector,variance_vector)
    plt.title('Explained Variance By N-Components')
#    plt.legend(['Test','Train'])
    plt.xlabel('Number of Components')
    plt.ylabel('Percent of Variance')
#    plt.ylim((0,1.0))
    plt.savefig('Titanic/PCA_explained_variance.png')
    plt.show()    

    '''
    #CHART: Eigenvalues
    print('eigenvalues vector',eigenvalues_vector)
    plt.plot(i_vector,eigenvalues_vector)
    plt.title('Eigenvalues By N-Components')
#    plt.legend(['Test','Train'])
    plt.xlabel('Number of Components')
    plt.ylabel('Eigenvalue')
#    plt.ylim((0,1.0))
    plt.savefig('Titanic/PCA_eigenvalues.png')
    plt.show()    
    '''    

    #CHART: Reconstruction Error/Loss Calc
    plt.plot(i_vector,loss_vector)
    plt.title('Reconstruction Error By N-Components')
#    plt.legend(['Test','Train'])
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
#    plt.ylim((0,1.0))
    plt.savefig('Titanic/PCA_Reconstruction_Error.png')
    plt.show()    

    
    pca=PCA(n_components=20)#10)
#    pca=PCA(.95) #@13:00 of video. get # PCs that contain 95% of variance
    pca.fit(X)
    print('Explained Variance Ratio',pca.explained_variance_ratio_)
    print('Sum EVR',sum(pca.explained_variance_ratio_))
#    print('Singular Values',pca.singular_values_)
    print('N components: ',pca.n_components)
    print('The components',pca.components_)
#    print('Columns',X.columns)
    #This applies the transform to your data. Now you can pump it into your model. It 'should' run faster now
    X_transformed = pca.fit_transform(X)
    #X_transformed = pca.transform(X)
    
    return X_transformed

def independent_components_analysis(data_set):
    #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
    print('------ICA------')
    X=data_set.copy()
    print('Columns',len(X.columns))

    kurtosis_vector=[]
    i_vector=[]
    loss_vector=[]
    i=1
    while i<30:#len(X.columns):
        transformer=FastICA(n_components=i,random_state=0)
        X_transformed=transformer.fit_transform(X)
        k_val=scipy.stats.kurtosis(X_transformed)
        k_val_ave=np.sum(np.sqrt(k_val*k_val))/i
        kurtosis_vector.append(k_val_ave)
        i_vector.append(i)
#        print('kurtosis',i,':',k_val)
        #kurtosis_vector=kurtosis_vector.append(k_val)       
        print('kurtosis AVE',i,':',k_val_ave)#kurtosis_vector=kurtosis_vector.append(k_val)       

        random_matrix=transformer.components_
        inv_matrix=np.linalg.pinv(random_matrix)
        X_inverse_transformed=np.dot(X_transformed,np.transpose(inv_matrix))
    
        loss_calc=((X_inverse_transformed-X)**2).sum()
        loss_calc=loss_calc.sum()
        loss_vector.append(loss_calc)

        i=i+1

    #10 Features Chosen
    transformer=FastICA(n_components=30,random_state=0)#n_components=10,random_state=0)
    X_transformed=transformer.fit_transform(X)
    
    print('shape',X_transformed.shape)
    print('kurtosis vector:',kurtosis_vector)
    print('components',transformer.components_)


    #Kurtosis Plot
    plt.plot(i_vector,kurtosis_vector)#(n_range,scores_list,n_range,train_scores_list)#n_range,scores_list)#,n_range,[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])#(['0.05','0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5'],scores_list)
#    plt.plot(n_range,scores_list,n_range,train_scores_list)
    plt.title('Kurtosis Values By N-Components')
#    plt.legend(['Test','Train'])
    plt.xlabel('Number of Components')
    plt.ylabel('Kurtosis')
#    plt.ylim((0,1.0))
    plt.savefig('Titanic/ICA_kurtosis.png')
    plt.show()    


    #CHART: Reconstruction Error/Loss Calc
    plt.plot(i_vector,loss_vector)
    plt.title('Reconstruction Error By N-Components')
#    plt.legend(['Test','Train'])
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
#    plt.ylim((0,1.0))
    plt.savefig('Titanic/RP_Reconstruction_Error.png')
    plt.show()    


    #NN Accuracy Plot

    return X_transformed

def random_projection(data_set):
    #https://scikit-learn.org/stable/modules/random_projection.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection
    #https://stackoverflow.com/questions/53134297/value-error-eps-0-100000-as-i-try-to-reduce-data-dimensionaity-what-could-be-th
    print('------RANDOM PROJECTION------')
    #which one to use?

    X=data_set.copy()
    

    i=1
    i_vector=[]
    loss_vector=[]
    while i<30:#len(X.columns):
        transformer=sklearn.random_projection.GaussianRandomProjection(n_components=i)#10)
        X_new=transformer.fit_transform(X)
        random_matrix=transformer.components_
        inv_matrix=np.linalg.pinv(random_matrix)
        X_inverse_transformed=np.dot(X_new,np.transpose(inv_matrix))
    
        loss_calc=((X_inverse_transformed-X)**2).sum()
        loss_calc=loss_calc.sum()
        i_vector.append(i)
        loss_vector.append(loss_calc)
        i=i+1


    #print('random matrix',random_matrix.shape)
#    print(X_new.shape)
#    print(X_inverse_transformed.shape)
#    print('loss',loss_calc)

    print('loss vector',loss_vector)

    #CHART: Reconstruction Error/Loss Calc
    plt.plot(i_vector,loss_vector)
    plt.title('Reconstruction Error By N-Components')
#    plt.legend(['Test','Train'])
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
#    plt.ylim((0,1.0))
    plt.savefig('Titanic/RP_Reconstruction_Error.png')
    plt.show()    

    print('RP Components:',transformer.components_)
    
    return X_new


def variance_threshold(data_set):
    #https://scikit-learn.org/stable/modules/random_projection.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection
    #https://stackoverflow.com/questions/53134297/value-error-eps-0-100000-as-i-try-to-reduce-data-dimensionaity-what-could-be-th
    print('------VARIANCE THRESHOLD------')
    #which one to use?

#    X=data_set.copy()
    
    min_max_scaler=preprocessing.MinMaxScaler()
    x_scaled=min_max_scaler.fit_transform(data_set.copy())
    X=pd.DataFrame(x_scaled)

    i=0
    i_vector=[]
    loss_vector=[]
    var_threshold_vector=[0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]#,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95]
    while i<len(var_threshold_vector):#30:#len(X.columns):
#        transformer=sklearn.random_projection.GaussianRandomProjection(n_components=i)#10)
        transformer=sklearn.feature_selection.VarianceThreshold(threshold=var_threshold_vector[i])#10)
        X_new=transformer.fit_transform(X)
        
#        random_matrix=transformer.components_
#        inv_matrix=np.linalg.pinv(random_matrix)
#        X_inverse_transformed=np.dot(X_new,np.transpose(inv_matrix))
        X_inverse_transformed=transformer.inverse_transform(X_new)
        loss_calc=((X_inverse_transformed-X)**2).sum()
        loss_calc=loss_calc.sum()
        i_vector.append(var_threshold_vector[i])#(i)
        loss_vector.append(loss_calc)
        print('Shape of Output for Threshold',var_threshold_vector[i],':',X_new.shape)
        i=i+1

    #print('random matrix',random_matrix.shape)
#    print(X_new.shape)
#    print(X_inverse_transformed.shape)
#    print('loss',loss_calc)

    print('loss vector',loss_vector)

    #CHART: Reconstruction Error/Loss Calc
    plt.plot(i_vector,loss_vector)
    plt.title('Reconstruction Error By K-Components')
#    plt.legend(['Test','Train'])
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
#    plt.ylim((0,1.0))
    plt.savefig('Titanic/RP_Reconstruction_Error.png')
    plt.show()    

    transformer=sklearn.feature_selection.VarianceThreshold(threshold=0.02)#10)
    X_new=transformer.fit_transform(X)

    X_new=pd.DataFrame(X_new)

#    print('Components Remaining:',X_new.columns)
    print('Components Remaining:',len(X_new.columns))
    
    return X_new

def neural_net(df_X,df_y):
    print("------ NEURAL NET ------")
    from sklearn.datasets import load_iris

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, \
                                                    test_size = 0.2, random_state = 3)
#    drop_list=['Survived','Name','Sex','Ticket','Cabin','PassengerId','Embarked']
#    drop_list2=['Name','Sex','Ticket','Cabin','PassengerId','Embarked']

#    X_train=df_train.drop(drop_list,axis=1)
#    X_test=df_test.drop(drop_list2,axis=1)
#    y_train=df_train['Survived']
#    y_test=df_submission['Survived']

#    X_train=principle_components_analysis(X_train)
#    X_test=principle_components_analysis(X_test)

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

    '''
    #PS1 MLPClassifier
    print('PS1 Accuracy')
    start_time=time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(13,13,13), activation='relu', max_iter=3000,learning_rate_init=0.0001, solver='adam')
    mlp.fit(X_train_scaled,y_train_hot)
    end_time=time.time()
    time_BackProp=end_time-start_time
    print('TIME BACKPROP',time_BackProp, end_time,start_time)
    predictions = mlp.predict(X_test_scaled)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
#    print(pd.crosstab(y_test,predictions,rownames=['True'],colnames=['Predicted'],margins=True))    
    '''

    #PS3 Classifier
    print('PS3 Accuracy')


    #GRIDSEARCH
    '''    
    parameters = { 'hidden_layer_sizes':[(13,13,13)],
                   'activation':['relu'],
                   'max_iter':[3000],
                   'learning_rate_init':[0.0001],
                   'solver':['adam']
        }

    mlpGS=MLPClassifier()
    clf=GridSearchCV(mlpGS,parameters)
    clf.fit(X_train_scaled,y_train_hot)
    sorted(clf.cv_results_.keys())
#    print(clf.cv_results_.keys())
    print('best estimator',clf.best_estimator_)
    print('best score',clf.best_score_)
    print('mean_train_score',clf.cv_results_['mean_train_score'])
    '''       

    start_time=time.time()
    mlp2 = MLPClassifier(hidden_layer_sizes=(13,13,13), activation='relu', max_iter=3000,learning_rate_init=0.0001, solver='adam')
    mlp2.fit(X_train_scaled,y_train_hot)
    end_time=time.time()
    time_RHC=end_time-start_time
    print('TIME BACKPROP',time_RHC, end_time,start_time)
#    nn_model1.fit(X_train_scaled, y_train)

    from sklearn.metrics import accuracy_score

    # Predict labels for train set and assess accuracy
    y_train_pred = mlp2.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
#    y_train_accuracy = accuracy_score(y_train, y_train)


#    print(y_train_pred,y_train_hot)
    print('training accuracy',y_train_accuracy)
    #0.45

    # Predict labels for test set and assess accuracy
    y_test_pred = mlp2.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
#    y_test_accuracy = accuracy_score(y_test, y_test)

    print('testing accuracy',y_test_accuracy)

    print(confusion_matrix(y_test_hot,y_test_pred))
    print(classification_report(y_test_hot,y_test_pred))
#    print(pd.crosstab(y_test_hot,y_test_pred,rownames=['True'],colnames=['Predicted'],margins=True))    

    '''
    #Time Charts
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
    '''

    return


if __name__ == "__main__": 			  		 			 	 	 		 		 	  		   	  			  	
    main()

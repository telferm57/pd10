# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:39:44 2017

@author: dad
"""
import logging, imp
import cfg, pdb
import pandas as pd
import synsigp as sp 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import datetime
import pickle
from synapse1 import setGlobals

imp.reload(cfg)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, log_loss 
min_max_scaler = preprocessing.MinMaxScaler()
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

# not used on voice  sp.featureList

# load into pd for clean-up
audioPath = 'audio\\audio_audio\\2\\features\\cleanedaudioFeatures.csv'
df = pd.read_csv(cfg.WRITE_PATH + audioPath)
df.columns
df.shape
df.isnull().sum()
df.healthCode.nunique()

#df.columns=sp.featureList - now in header 
# map diagnosoed to 1/0 
df.diagnosed = df.diagnosed.map({True:1,False:0})
df['gender'].hist()
df['updrs2_1'].hist()
df['diagnosed'].hist()
# df.drop(['healthCode','diagnosed'],axis=1,inplace=True) # for presicting updrs score 
# df.drop(['healthCode','updrs2_1'],axis=1,inplace=True) # for predicting diagnosed 
df.columns
#==============================================================================
# ii = 0  # last field ni jotter was /XLA 
# ij = 0 
# for g in df['jitter_x']:
#     ii += 1
#     try:
#         tt = float(g)
#     except:
#         print(g,' not floatable')
#         ij += 1
#     print(ii,ij)  
#==============================================================================
    
    
yy = df.diagnosed.values
yy = np.ravel(yy)
yy[1:10]
z = list(map(lambda x:float(x),yy))
y = z
len(y)
X = df.drop(['diagnosed','updrs2_1'],axis=1) # will remove healthcode after test train split
features = X.columns.values

X = X.values
X[0,20:]
X.shape

#%% multiple classifiers - find the best 
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#
#-----------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing



names = ["Nearest Neighbors", "Linear SVM", 
         #"RxdcBF SVM",
         "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
names_t = ["Nearest Neighbors", "Linear SVM"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

classifiers_t = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025)
    ]
results_mcc = {} 
results_cm = {}
results_cv ={} 
all_results = {}


def calcScores(df,series,joinClasses,target,clfNames,classifiers,
               clfName=0,varT=0): 
    
    ''' clfName - to limit scores to one clf  
        varT - low variance threshold '''
    X, y, features = splitFeatureDF(df,excludeFields[series],joinClasses,target)
    #todo remove collinear (see below - manual at the moment)
    Xlow, selectedLowVar = removeLowVariance(X,varT) # 0,015 for walk reduces to 30 - got by experimentation
    Xlow, importances = drExtraTrees(X, y) # use extratrees to get most important feats 
    print('number of features femoved with var < ',varT,X[0].size - Xlow[0].size)
    print(pd.Series(y).value_counts())
        # compare each classifier on same split 
    X_train, X_test, y_train, y_test = train_test_split(Xlow, y, test_size=.3)
    
    for name, clf in zip(clfNames, classifiers):
        if clfName:
            if name != clfName: continue
            
        print('processing :',name)   
        clf.set_params(**prf1[name])
        if name == 'svc': clf.set_params(probability=True)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        y_pred= clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        len(y_proba)
        y_favoured = getFavouredClass(y_pred,y_proba)
        y_ord = getOrdClass(clf,X_train, X_test,y_train,y_test)
        y_ord.shape
        
        favhist = np.histogram(y_favoured,bins=[-1,0,1,2,3,4])
        predhist = np.histogram(y_pred,bins=[-1,0,1,2,3,4])
        ordhist = np.histogram(y_ord,bins=[-1,0,1,2,3,4])
      #  for a in y_proba[:10]:print(a)
      #  y_favoured[:10]
        precision, recall, fscore, support = scoreAll(y_test, y_pred)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
        f1_macro=f1_score(y_test, y_pred,average='macro')
        f1_macro_fav =f1_score(y_test, y_favoured, average='macro')
        f1_macro_ord =f1_score(y_test, y_ord, average='macro')
        mad_rate = sum(abs(y_pred-y_test))/len(y_test) # mean absolute deviation
        mad_rate_f = sum(abs(y_favoured-y_test))/len(y_test)
        mad_rate_o = sum(abs(y_ord-y_test))/len(y_test)
        # normailsed mad rate ? at the moment favouring populous classes 
        #df_test.shape
        #y_pred_hc, y_test_hc = majorityScore(y_pred,df_test)
        cm = confusion_matrix(y_test,y_pred)
        cmf = confusion_matrix(y_test,y_favoured)
        cmo = confusion_matrix(y_test,y_ord)
        logloss = log_loss(y_test,y_proba)
    
        cm_hc = None # only for multiple records per healthCode 

   
       # cm_hc = confusion_matrix(y_test_hc,y_pred_hc)
    #len(y_pred)
    #results_mcc[name] = cross_val_score(clf, item, y, cv=5,scoring=mc_scorer)
        timeNow = datetime.datetime.now().isoformat(' ', 'seconds') 
        if name in all_series_results.keys():
            all_series_results[name].update({series:[timeNow, logloss, cm,
                              cmf,cmo,mad_rate,mad_rate_f,mad_rate_o]})
         
        else: #new algorithm - add to results dict 
            all_series_results[name] = {series:[timeNow, logloss,f1_macro, cm, 
                              cmf,cmo,mad_rate,mad_rate_f,mad_rate_o]}            
        pickle.dump(all_series_results,
                    open('all_results_series_pickle_' + series,"wb"))   
        print('Completed :',name, 'time:', timeNow)   
    

#%% experiment with feature selection 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
#%% feature selection - extra tree 

clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
model = SelectFromModel(clf,prefit=True,threshold=0.03)
X_new = model.transform(X)
X_new.shape 
X.shape        
clf.feature_importances_  
list(zip(features,clf.feature_importances_  ))      

#%% feature selection - lsvc 
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape
X.shape
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)
#%%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() /1.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#%%
# 
#random forest , SVC, knn 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors
from sklearn.neural_network import MLPClassifier

from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC

# great for acceleromter : c = RandomForestClassifier()
c = SVC(kernel="linear", gamma=0.005,C=1,probability=True)
#c = svm.SVC()
n_neighbors = 5 
weights='uniform'
#c = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
#c = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                    hidden_layer_sizes=(5, 2), random_state=1)
b = DummyClassifier() # generates predictions by respecting the training set's class distribution

results = []
baselines = []
feature_importancy =[]
X_minmax[0]
for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    # assuming Z still had healthcode, save test and train hc values for later combination
    # with walking data 
    X_Tr_hc = X_train[:,-1] 
    X_Ts_hc = X_test[:,-1] 
    X_test = X_test[:,0:-1]
    X_train = X_train[:,0:-1]
  
    c.fit(X_train, y_train)
    b.fit(X_train, y_train)
    res = c.score(X_test, y_test)
    bas = b.score(X_test, y_test)
    print('Loop', 'i', res, bas)
   # print(c.feature_importancies_)
    results.append(res)
    baselines.append(bas)

print( '\nBaseline', np.mean(baselines), np.std(baselines))
print( 'Random Forest', np.mean(results), np.std(results))
#%% get predictions as probabilities
# run prediction on whole dataset 
X_all = X[:,0:-1] # remove healthcode
hc_all = X[:,-1]
hc_all.shape
Y_pred_prob = c.predict_proba(X_all)
type(Y_pred_prob)
Y_pred_prob[1:20]
len(Y_pred_prob)
len(hc_all)
audio_pwp = pd.DataFrame(Y_pred_prob.copy())
audio_pwp.describe()
audio_pwp['healthCode'] = hc_all
audio_pwp.to_csv(cfg.WRITE_PATH + audiofPath + 'probdiag',index=False)
# attach to a healthcode - we need to do our own split beforehand - or include healthcode in data at weight 0 ?  

#%% tune SVC ... bet it makes no diff ! 
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf','poly'), 'C':[0.1,0.5,1,2,6,10],
              'gamma':[0.005,0.01,0.03,0.05,0.1]}
svr = SVC()
from sklearn.metrics import matthews_corrcoef,fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)
clf = GridSearchCV(svr, parameters)
clf.fit(X_minmax, y)
clf.cv_results_
clf.cv_results_['params'][clf.best_index_]
#%%
y_pred= c.predict(X_test)
len(y_pred)
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef 

matthews_corrcoef(y_test, y_pred) 

cm = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)
#%%
# Plot non-normalized confusion matrix
plt.figure()
#plot_confusion_matrix(cm, classes=[-1,0,1,2,3,4],normalize=True,
plot_confusion_matrix(cm, classes=[0,1],normalize=True,
                      title='Confusion matrix, with normalization')
plt.figure()
plot_confusion_matrix(cm, classes=[0,1],normalize=False,
                      title='Confusion matrix, without normalization')
plt.show()
#%%

squares = list(map(lambda x: x**2, range(10)))
print(squares)

import importlib as imp
imp.reload(ap)

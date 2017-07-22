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
from sklearn.metrics import confusion_matrix
imp.reload(sp)
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
from sklearn.metrics import confusion_matrix
min_max_scaler = preprocessing.MinMaxScaler()

#==============================================================================
# type(sp.featureList)
# tt = sp.featureList.copy()
# 
# for feature in excludeFields:
#     print(feature)
#     features = tt.remove(feature)
# features = tt.copy()
#==============================================================================

# load into pd for clean-up
def loadWalkData():
    df = pd.read_csv(cfg.WRITE_PATH + cfg.WALK_PATH +'combinedfeatures.csv')
    df.columns=sp.featureList 

    df.diagnosed = df.diagnosed.map({'True':1,'False':0})
    df['gender']=df['gender'].map({'Male':1,'Female':0})
    lb4 = df.shape[0] 
    df.dropna(inplace=True)
    droppedRows = lb4 - df.shape[0]
    return df, droppedRows

def splitByHealthCode(df,excludeFields,frac=0.7):
    ''' this is only relevant where there are multiple records per person, 
    ensuring that all records for one individual are assigned to either training
    or test, NOT BOTH!
    for one record per person , splitFetaureDF '''
   # excludeFields = ['healthCode','diagnosed','audioProb','updrs2_12']
   # excludeFields = ['healthCode','diagnosed','updrs2_12']
    hc = df.healthCode.unique()
    hc[:10]
    hc.shape
    #split healthcode randomly in proportion 70:30 
    x_train_len = int(np.floor(len(hc)*frac))
    x_test_len = len(hc) - x_train_len
    hc_train,hc_test = np.split(np.random.permutation(hc),[x_train_len])
    hc_test[:10]
    # now use these indices to split the df into 2 and construct test and train from those 
    df_train = df[df.healthCode.isin(hc_train)]
    df_train = df_train.sample(frac=1)
    df_train.shape
    df_train.columns
    df_test = df[df.healthCode.isin(hc_test)]
    df_test = df_test.sample(frac=1)
    df_test.shape
    df_test.columns
    #check df_train and test match up to the df they were extracted from
    #TODO turn these into assertions
    f10 = df_train.index[1:1000].values
    (df_train.loc[f10].updrs2_12 != df.loc[f10].updrs2_12).sum()
    (df_train.loc[f10].healthCode != df.loc[f10].healthCode).sum()
    f11 = df_test.index[1:1000].values
    (df_test.loc[f11].updrs2_12 != df.loc[f11].updrs2_12).sum()
    # in desperation lets randomly drop 30% of df_train (this is basically what happens in
    # rendomly selecting 70%) -note we have already shuffled df_train
    #df_train = df_train.sample(frac=0.7)
    # made no difference - acc still in the 30% region ! 
    len(df_test) + len(df_train) == len(df)
    # construct X_train and y_train and test
    X_train = df_train.drop(excludeFields,axis=1)
    X_train = X_train.values
    # are these in the same order ? 
    (X_train[0:100,1:2] !=df_train.iloc[0:100,1:2].values).sum()
    df_test.head()
    y_train = np.ravel(df_train.updrs2_12)
    len(y_train)
    len(X_train)
    X_test = df_test.drop(excludeFields,axis=1)
    print('xtest cols after drop ',X_test.columns)
    X_test = X_test.values
    y_test = np.ravel(df_test.updrs2_12)
    # do the first 100 values in y_train match the first 100 values of updrs2_12 in
    # df_train ? 
    sum(y_train[0:100] !=df_train.iloc[0:100,60])==0
    # len(y_test)
    #len(X_test)
    # und zo all values as expected 
    min_max_scaler = preprocessing.MinMaxScaler()
    X_test_minmax = min_max_scaler.fit_transform(X_test)
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    
    #also contruct as X, Y for comparison with above method 
    #X = df.drop(['healthCode','diagnosed','updrs2_12'],axis=1)
    #X = X.values
    #y = np.ravel(df.updrs2_12)
    
    return X_train_minmax, X_test_minmax, y_train, y_test, df_train, df_test

def splitFeatureDF(df,excludeFields,joinClasses=False,target='updrs2_12',frac=0.7):
    ''' split feature dataframe into X and Y 
    exclude fields typically ['healthCode','diagnosed','audioProb','updrs2_12']
    - target is normally the updrs code '''
    dfthis = df.copy()
    excludeFields = ['healthCode','diagnosed','audioProb','updrs2_12']
    if  joinClasses:
        #joinClasses ={2:1,3:2,4:2} # map 1,2 to 1, 3,4 to 2 
        dfthis[target]=dfthis[target].map(joinClasses)
    
    X = dfthis.drop(excludeFields,axis=1)
    X.shape
    X = X.values

    y = np.ravel(dfthis[target])
  
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    X_unscaled = X.copy()
    X = X_minmax.copy()
    
    return X, y 

#%%

df, droppedRows = loadWalkData()
joinClasses ={-1:-1,0:0,1:1,2:1,3:2,4:2} # map 1,2 to 1, 3,4 to 2 
joinClasses = False
df.updrs2_12.value_counts()
p, j = splitFeatureDF(df,excludeFields,joinClasses)

pd.Series(j).value_counts()
df.updrs2_12.hist()
p.shape
df.shape
Xlow = removeLowVariance(p,0.015) # reduces to 30 - got by experimentation
Xlow.shape
y=j.copy


pickle.dump(Xlow,open('Xlow_19_07_17',"wb"))  
pickle.dump(X,open('X_19_07_17',"wb"))  
pickle.dump(y,open('X_19_07_17',"wb")) 


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
#%% get majority score for test data set 
def majorityScore(y_pred,df_test, target='MDS-UPDRS2.12'):
    '''Pass in a predicted score set for X_test, I will return
    a list of healthcodes and respective predicted updrs scores based on the
    majority score ''' 
    #X_text is drawn from df_test which has the health code - it is in the 
    # same order, so we can just add the prediction in 
    print('len ypred passed in:', len(y_pred),'len df_test I can See', len(df_test))
    df_test['y_pred'] = y_pred
    #for each healthcode, get the majority value of ypred
    df_test_frequency = df_test.groupby(
            ['healthCode','y_pred']).size().unstack(fill_value=0) 
    
    df_test_frequency['pred'] = df_test_frequency.idxmax(axis=1)
    df_test_frequency['pred'].hist()
    # df_test_frequency now contains the predicted values in 'pred' - need to score
    #  against the real values ...which are in the scores table  
    scorescut = scores[['healthCode',target]]
    #join the real scores to the max-vote scores from prediction 
    joined = df_test_frequency.join(scorescut.set_index('healthCode'))
    # pwithoutp will not have a updrs score, and appear as nan -  change to -1, 
    # our indictor for person without parkinsons 
    joined['MDS-UPDRS2.12'].replace(to_replace=np.nan,value=-1,inplace=True)
   
    y_pred_majority = joined['pred']
    y_test_hc = joined['MDS-UPDRS2.12']
    return y_pred_majority, y_test_hc    
    
#%% test multiple classifiers 
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
from sklearn.metrics import confusion_matrix
min_max_scaler = preprocessing.MinMaxScaler()


clfNames = ["Nearest Neighbors", "Linear SVM", 
         #"RBF SVM",
       #  "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
clfNames_t = ["Nearest Neighbors","Linear SVM","Gaussian Process"]
clfNames_2 = ["Gaussian Process"]
clfNames_2 = ["Decision Tree", "Random Forest"]
clfNames_2 = ["Naive Bayes", "QDA"]
classifiers = [
    KNeighborsClassifier(algorithm='brute',n_neighbors=5,p= 1, weights='distance'),
    SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(criterion='entropy',max_depth=10,
                           min_samples_leaf=5, min_samples_split=10),
   
    RandomForestClassifier(max_depth=10,max_features=17, n_estimators=10),
    MLPClassifier(alpha=1, 
                  activation='tanh',hidden_layer_sizes=(150, 97, 69, 29),
                  learning_rate='adaptive', solver='lbfgs'),
    AdaBoostClassifier(learning_rate=0.6,n_estimators=70),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

classifiers_t = [KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
    ]
classifiers_2 = [
        GaussianNB(),
    QuadraticDiscriminantAnalysis()
    ]




#datasets = {'X':X,'minmax':X_minmax,'reduced':X_new}
datasets ={} # only one hardcoded now 

from sklearn.metrics import confusion_matrix,f1_score    
from sklearn.metrics import matthews_corrcoef,fbeta_score, make_scorer
from sklearn.metrics import precision_recall_fscore_support as scoreAll
ftwo_scorer = make_scorer(fbeta_score, beta=2)
from sklearn.model_selection import cross_val_score
mc_scorer = make_scorer(matthews_corrcoef)

import pickle

clf = GaussianNB()
# del all_results
results_mcc = {} 
results_cm = {}
results_cv ={} 
all_results = {}
all_series_results = {}
excludeFields={}

seriesDescription = {'s0':'No prediction of diagnosis using audio',
                      's1':'audio diagnosis added',
                      's6':'tuned rf, lowminmax',
                      's7':'tuned rf, lowminmax, include Audio',
                      's8':'all clf tuned: lowminmax:exclude Audio',
                      's9':'all clf tuned: lowminmax:include Audio',
                      's10':'all clf tuned: lowminmax:include Audioannd diagnosed',
                      's11':'all clf tuned:low var excl:classes 1,2;3,4 joined',
                      's12':'s11 + audio',
                      's13':'deduped dataset',    
                      's2':'real diagnosis added',
                      's3':'new gait data and mean of signals, no feature selection',
                      's5':'new gait data and m sigs,low var and corr vars removed',
                      's4':'low variance removed'}
excludeFields['s0'] = ['healthCode','diagnosed','audioProb','updrs2_12']
excludeFields['s2'] = ['healthCode','audioProb','updrs2_12']
excludeFields['s3'] = ['healthCode','diagnosed','audioProb','updrs2_12']
excludeFields['s4'] = ['healthCode','diagnosed','audioProb','updrs2_12']
excludeFields['s5'] = ['healthCode','diagnosed','audioProb','updrs2_12']
excludeFields['s6'] = ['healthCode','diagnosed','audioProb','updrs2_12']
excludeFields['s7'] = ['healthCode','diagnosed','updrs2_12']
excludeFields['s8'] = ['healthCode','diagnosed','audioProb','updrs2_12']
excludeFields['s9'] = ['healthCode','diagnosed','updrs2_12']
excludeFields['s10'] = ['healthCode','updrs2_12']
excludeFields['s11'] = ['healthCode','diagnosed','audioProb','updrs2_12']
excludeFields['s12'] = ['healthCode','diagnosed','updrs2_12'],
excludeFields['s13'] = ['healthCode','diagnosed','audioProb','updrs2_12']
# -----------------------------------------------------------------------------

def calcScores(df,series,joinClasses): 
    #create datasets based on exclude fields 
    # X_train, X_test, y_train, y_test, df_train, \
        #    df_test = splitByHealthCode(df,excludeFields,0.7)
    #
    
    X, y = splitFeatureDF(df,excludeFields[series],joinClasses)
    #todo remove collinear (see below - manual at the moment)
    Xlow = removeLowVariance(X,0.015) # reduces to 30 - got by experimentation
    print(pd.Series(y).value_counts())
        # compare each classifier on same split 
    X_train, X_test, y_train, y_test = train_test_split(Xlow, y, test_size=.3)
    
    for name, clf in zip(clfNames, classifiers):
        print('processing :',name)       
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        y_pred= clf.predict(X_test)
        precision, recall, fscore, support = scoreAll(y_test, y_pred)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
        f1_macro=f1_score(y_test, y_pred,average='macro')
        
  
        #df_test.shape
        #y_pred_hc, y_test_hc = majorityScore(y_pred,df_test)
        cm = confusion_matrix(y_test,y_pred)
        
        cm_hc = None # only for multiple records per healthCode 

   
       # cm_hc = confusion_matrix(y_test_hc,y_pred_hc)
    #len(y_pred)
    #results_mcc[name] = cross_val_score(clf, item, y, cv=5,scoring=mc_scorer)
        timeNow = datetime.datetime.now().isoformat(' ', 'seconds') 
        if name in all_series_results.keys():
            all_series_results[name].update({series:[timeNow, score, cm, cm_hc]})
         
        else: #new algorithm - add to results dict 
            all_series_results[name] = {series:[timeNow, score,f1_macro, cm, cm_hc]}            
        pickle.dump(all_series_results,
                    open('all_results_series_pickle_' + series,"wb"))   
        print('Completed :',name, 'time:', timeNow)   
        
#%% -----------------------------------------------------------------------------

# make sure scores (from the getScores function that reads the scores dataset)
#  are loaded before this ! 
df, droppedRows = loadWalkData()
joinClasses ={-1:-1,0:0,1:1,2:1,3:2,4:2} # map 1,2 to 1, 3,4 to 2 
joinClasses = False
calcScores(df,'s13',joinClasses) 

all_res  = pickle.load(open('all_results_series_pickle_s4', "rb")) 
# saving raw results  pickle.dump(all_res,open('all_results_pickle_1', "wb")) 
y_pred.shape  
all_series_results = {}

for key, value in all_results.items():
  
    if key in all_series_results.keys(): 
        all_series_results[key].update({value[1]: [value[0]]+ value[2:]})  
    else:
        all_series_results[key] = ({value[1]: [value[0]]+ value[2:]}) 
        
for key, value in all_res.items():
    
    if key in all_series_results.keys(): 
        all_series_results[key].update(value)  
    else:
        all_series_results[key] = (value) 
  

# ---------------------------------------------------------------------------
def printSeriesResults(series,classes):
    for name in all_series_results:
        #print(all_results[name][1])
       desc = seriesDescription[series]
       print(name,'   ',desc)
       print('f1_macro score: %.3f'%all_series_results[name][series][1])
       plot_confusion_matrix(all_series_results[name][series][2],
                              title=name + ' ' + desc,classes=classes,
                              normalize=True)
# ---------------------------------------------------------------------------       
   def printSeriesF1macro(series):
       print(seriesDescription[series])
       for name in all_series_results:
            #print(all_results[name][1])
           print('%15s : f1_macro score: %.3f'%(name, all_series_results[name][series][1]))

classes6 = [-1,0,1,2,3,4]
classes4 = [-1,0,1,2]   
printSeriesResults('s12',classes4) 
printSeriesResults('s13',classes6) 
printSeriesF1macro('s6')
printSeriesF1macro('s8')
printSeriesF1macro('s9')
printSeriesF1macro('s10')
printSeriesF1macro('s11')
printSeriesF1macro('s13')
    
    plot_confusion_matrix(all_series_results[name][series][3],
                          title=name + ' ' + desc+' by health code',classes=[-1,0,1,2,3,4],
                          normalize=True)
# print scores    
for name, value in all_series_results.items():
    for series, values in value.items():
        print (name, series, values[1])
        
   desc = seriesDescription[series]
   plot_confusion_matrix(all_series_results[name][series][2],
                          title=name + ' ' + desc,classes=[-1,0,1,2,3,4],
                          normalize=True)
   plot_confusion_matrix(all_series_results[name][series][3],
                          title=name + ' ' + desc+' by health code',classes=[-1,0,1,2,3,4],
                          normalize=True)
   
    

all_results[0]

with open('all_results_pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    tAR = pickle.load(f)

tAR["Nearest Neighbors"][0]
calcScores(datasets) 

for key, value  in all_results.items():
    print(key)
    for key2, value2 in value.items():
       # print(value2)
       # print(value2.mean())
        print(' %15s mean %1.2f std %1.2f'% (key2,value2.mean(),value2.std()))
    

#%% experiment with feature selection 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
#%% feature selection - extra tree 
# low variance 
def removeLowVariance(X,thresh):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=thresh)
    sel.fit(X)
    selectedLowVar = sel.get_support(indices=True)
    Xlow = sel.fit_transform(X)
    Xlow.shape
    return Xlow

def removeCollinear(X):
    '''remove collinear vars - manual at the moemnt  '''

    import numpy as np
    # check for collinearity
    # remove cols with 0 std 
    idx =[]
    m = np.zeros_like(Xlow)
    
    for i in range(Xlow.shape[1]):
        if np.std(Xlow[:,i], axis=0)==0:
          idx.append(i)
    ix = np.array(idx)
    
    tt = np.delete(Xlow,ix,1)
    
    corr = np.corrcoef(tt, rowvar=0)
    corr  # correlation matrix
    w, v = np.linalg.eig(corr)
    #check eigenvalues near 0
    inc = 0
    
    for dd in w:
     
        #print(dd)
        if dd<0.01:
            print('{:.10f}'.format(dd))
            print(inc)
        inc += 1
    
    #17 & 24 have low eigenvalues 
    v[:,17].tolist()
    [i for i,v in enumerate(v[:,17]) if abs(v) >  0.5]
    
    [i for i,v in enumerate(v[:,24]) if abs(v) >  0.5]
    
    # it seems 5 and 7 and 17 and 20 are collinear - what are these ? 
    featuresArray = np.array(features)
    featuresArray[[selectedLowVar[[5,7,17,20]]]]
    featuresArray[[selectedLowVar]].tolist()
    # drop 7 and 17 
    Xlow.shape
    tt = np.delete(Xlow,[7,17],1)
    tt.shape
    Xlow = tt.copy()
# Xlow has all low variance and crrelated variables removed 
#%% extratrees for iding low feature importances - not in use at the moment 
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
model = SelectFromModel(clf,prefit=True,threshold=0.03)
X_new = model.transform(X)
X_new.shape 
X.shape        
clf.feature_importances_  
list(zip(features,clf.feature_importances_  ))  
# graphically 
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(15,15))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()    

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
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
        print("Normalized confusion matrix")
        cmap=plt.cm.Blues
    else:
        print('Confusion matrix, without normalization')
        
    np.set_printoptions(precision=2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    

    print(cm)

    thresh = cm.max() /1.2
    if normalize: thresh = 0.80
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.close()
#%%
# 
#random forest , SVC, knn 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC


c = SVC(kernel="linear", gamma=0.005,C=1)
c = RandomForestClassifier(max_features=30)
c = RandomForestClassifier()
c = KNeighborsClassifier(5)
#c = svm.SVC()
n_neighbors = 5 
weights='uniform'
c = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
#c = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                    hidden_layer_sizes=(5, 2), random_state=1)
b = DummyClassifier() # generates predictions by respecting the training set's class distribution

results = []
baselines = []
feature_importancy =[]
print(X_train.shape,  y_train.shape)
y_train[1:10]
y_test[1:10]
y_train = y_train.astype(int)
#==============================================================================
y_test = y_test.astype(int)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
c.fit(X_train_minmax, y_train)
b.fit(X_train_minmax, y_train)
res = c.score(X_test_minmax, y_test)
bas = b.score(X_test_minmax, y_test)
print('Loop', res, bas)
#==============================================================================
 
#==============================================================================
for i in range(0, 10):
    
    # call bespoke test train split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    c.fit(X_train, y_train)
    b.fit(X_train, y_train)
    res = c.score(X_test, y_test)
    bas = b.score(X_test, y_test)
    print('Loop', i, res, bas)
    # print(c.feature_importancies_)
    results.append(res)
    baselines.append(bas)
# 
#==============================================================================
print( '\nBaseline', np.mean(baselines), np.std(baselines))
print( 'Random Forest', np.mean(results), np.std(results))
#%% tune random tree  ... bet it makes no diff ! Oh yes it does ! max_features changes everything
# tune promising models : adaboost % : )
tuningResults = {}    
from sklearn.model_selection import GridSearchCV

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=10,max_features=17, n_estimators=10),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

parameters = [
                {'n_neighbors':[3,5,7],'weights':['uniform','distance'],
                 'algorithm':['brute','ball_tree','kd_tree'],'p':[1,2]},
                 [  {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'C': [1, 10, 100, 1000]}
                 ],
              {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              },
              {'max_features':[10,12,14,17],'max_depth':[2,4,6,8,10],
              'n_estimators':[10,15,17,20]},
              {'hidden_layer_sizes':[(150,97,69,29),(200,100,37,15),(277,42,13)],
                                     'activation':['tanh','relu'],
                                     'solver':['lbfgs','adam'],
                                     'learning_rate':['constant','adaptive'],
                                     'learning_rate_init':[0.05, 0.01, 0.005, 0.001]},
              {'n_estimators':[30,40,50,60,70],'learning_rate':[0.5,0.6,0.7,0.8,1]},
              {'GuassianNB - only priors'},
              {'QDA - no hyperparameters'},
               ]


from sklearn.metrics import matthews_corrcoef,fbeta_score,precision_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)
precision_macro = make_scorer(precision_score, average='macro')
f1_macro=f1_score(y_test, y_pred,average='macro')
classif = 2

clf = GridSearchCV(classifiers[classif], parameters[classif])
clf.fit(Xlow_minmax, y)
clf.cv_results_
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
tuningResults[clfNames[classif]] = clf.cv_results_['params'][clf.best_index_]
pickle.dump(tuningResults,open('tuningResults','wb'))

#%%
y_pred= c.predict(X_test)
len(y_pred)
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef 

#matthews_corrcoef(y_test, y_pred) 

cm = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)
#%%
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=[-1,0,1,2,3,4],normalize=True,
#plot_confusion_matrix(cm, classes=[0,1],normalize=True,
                      title='Confusion matrix, with normalization')
plt.figure()
plot_confusion_matrix(cm, classes=[-1,0,1,2,3,4],normalize=False,
                      title='Confusion matrix, without normalization')
plt.show()
#%% 

squares = list(map(lambda x: x**2, range(10)))
print(squares)
scores['MDS-UPDRS2.12'].value_counts()

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
import gatherdata as syn1
from gatherdata import setGlobals

#imp.reload(synapse1)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
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
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import confusion_matrix, log_loss, f1_score,accuracy_score
min_max_scaler = preprocessing.MinMaxScaler()
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_recall_fscore_support as scoreAll
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import fbeta_score, precision_score, make_scorer
from sklearn.dummy import DummyClassifier
global all_series_results2,seriesDescription 
seriesDescription = {}
results_mcc = {} 
results_cm = {}
results_cv ={} 
all_results = {}
all_series_results = {}
excludeFields={}
all_series_results2 = {}
remcl = {} 
seriesFeatures = {}

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
def loadWalkData(rest=False,runNo=1):
    ''' walk data can be either walk data or standing (rest) data ''' 
    global restFeatureCols
    if rest:
       # df = pd.read_csv(cfg.WRITE_PATH + cfg.REST_PATH +'combinedfeatures_withg.csv',
        #                 header=0)
        df = pd.read_csv(cfg.WRITE_PATH + cfg.REST_PATH + str(runNo)  +'\\restcombined.csv',
                         dtype={'professionalDiagnosis':np.bool},#header=0)
                                                header=None)
        df.columns=restFeatureCols
       # df.columns = restFeatureCols # restFeatureCols is global 
       
        df.rename(columns={0:'age',1:'gender',2:'professionalDiagnosis',3:'onset_year',
                           4:'updrs2_1',5:'updrs2_10',
                6:'updrs2_12',7:'updrs2_13',8:'healthCode'},inplace=True)
        df.rename(columns={'professionalDiagnosis':'diagnosed'},inplace=True)
    else:    
        df = pd.read_csv(cfg.WRITE_PATH + cfg.WALK_PATH + str(runNo)  +'\\combinedfeatures.csv',
                         header=None)
        audioProbs=False
        features = sp.featureList.copy()
        if not audioProbs: features.remove('audioProb')
        df.columns=features
        
    df.isnull().sum() 
    try:
        df.audioProb = df.audioProb.fillna(value=0.5) # i.e. 50:50
    except:
        pass
    try:
        df.onset_year = df.onset_year.fillna(value=0) # i.e. 50:50
    except: pass
        #no audioprob
    df['diagnosed'] = df.diagnosed.map({True:1,False:0}) # nb true/false are strings here 
    df['gender']=df['gender'].map({'Male':1,'Female':0})
    lb4 = df.shape[0] 
    df.dropna(inplace=True)
    droppedRows = lb4 - df.shape[0]
    print('shape ', df.shape,' dropped rows: ',droppedRows )
    print(df.updrs2_1.value_counts())
     
    return df, droppedRows

def loadVoiceData(runNo):
    ''' note voice data is cleaned in preceding steps (no na's) and has a header 
    ''' 
    
    audioPath = 'audio\\audio_audio\\' + str(runNo) + '\\features\\cleanedaudioFeatures.csv'
    df = pd.read_csv(cfg.WRITE_PATH + audioPath)
    
    df.diagnosed = df.diagnosed.map({True:1,False:0})
    #df['gender']=df['gender'].map({'Male':1,'Female':0})
    droppedRows = 0
    print('shape ', df.shape)
    print(df.updrs2_1.value_counts())
    print(df.isnull().sum())
    # onehot encode medpoint 
    mp = df.medPoint.values
    mp = mp.reshape(-1,1)
    enc = OneHotEncoder()
    mp1hot = enc.fit_transform(mp)
    df2 =  pd.get_dummies(df,prefix=None,columns=['medPoint'])
    # df2.drop('medPoint_0.716524216524',inplace=True)
    df2.drop(df2.columns[-4],axis =1,inplace=True)
    
    return df2, droppedRows

def voiceDataStats(df):
    mpc = df.medPoint.value_counts()
    type(mpc)
    medPointmap =  {0:"I don't take Parkinson medications",
                3:"Immediately before Parkinson medication",
                1:"Just after Parkinson medication (at your best)",
                2:"Another time"}
    # graph distribution 
    fig, ax = plt.subplots()
    a_heights, a_bins = np.histogram(df.medPoint,bins=[0,1,2,3,4])
#b_heights, b_bins = np.histogram(dfd['MDS-UPDRS2.12'], bins=a_bins)
#c_heights, c_bins = np.histogram(dfd['MDS-UPDRS2.10'], bins=b_bins)
    width = (a_bins[1] - a_bins[0])/2

    ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
#ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen')
#ax.bar(c_bins[:-1]+2*width, c_heights, width=width, facecolor='yellow')
    plt.show()
    # of those 'just after' , how close is datafile ? 
    
    return

def verifyVoiceData():
    ''' veryfying voice data characteristics 
    1. does age match age in demogs ? 
    2. is there at least 1 score that matches the updrs scores from df? 
    TODO: 
    3. is there a balance of  nonPwp's PwPs?
    4. is the age profile balanced ? 
    ''' 
    def demogage(hc):
        return demogs[demogs.healthCode==hc].age.iloc[0]
    
    def updrscheck(row):
        #rhealthCode = '7b9b9932-4d19-49dd-8893-0441eba9ad46'
        if row.updrs2_1 == -1: return 0
        match = scores[(scores.healthCode==row.healthCode) & 
                       (scores['MDS-UPDRS2.1'] == row.updrs2_1) &
                       (scores['MDS-UPDRS2.12']== row.updrs2_12) ]
        if match.shape[0]==0: # cannot find matching record 
            print(row.healthCode)
            return 1
        else: return 0
            
        
    
    df['demogage'] = df.healthCode.apply(demogage) 
    assert sum(df.demogage!=df.age)==0
    
    pp = df.apply(updrscheck,axis=1) 
    assert sum(pp) == 0
    
    return

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

def getOutlier(df,sds):
    ''' reurns index of all rows with 1 column > sds sd's from mean '''
    from scipy import stats
    df.columns
    numericfeatures = ['meanPitch','medianPitch', 'meanPitch', 'sdPitch', 'minPitch', 'maxPitch',
       'nPulses', 'nPeriods', 'meanPeriod', 'sdPeriod', 'pctUnvoiced',
       'nVoicebreaks', 'pctVoicebreaks', 'jitter_loc', 'jitter_loc_abs',
       'jitter_rap', 'jitter_ppq5', 'shimmer_loc', 'shimmer_loc.1',
       'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'mean_autocor',
       'mean_nhr', 'mean_hnr']
    rowset = set()
    for f in numericfeatures:

        outliers = df[f][np.abs(stats.zscore(df[f]))>sds].index.values
     
        rowset = rowset.union(set(outliers))
        
        print('f :',f,':',
              df[f][np.abs(stats.zscore(df[f]))>sds].shape[0])
    return rowset

   
def splitFeatureDF(df,excludeFields,joinClasses,removeClasses,
                   target='updrs2_12',frac=0.7):
    ''' split feature dataframe into X and Y. n.b. Target dropped from X. 
    exclude fields typically ['healthCode','diagnosed','audioProb','updrs2_12']
    - target is normally the updrs code ; apply minmax scalar '''

    dfthis = df.copy()
    #excludeFields = ['healthCode','diagnosed','audioProb','updrs2_12']
    if  joinClasses:
        #joinClasses ={2:1,3:2,4:2} # map 1,2 to 1, 3,4 to 2 
        dfthis[target]=dfthis[target].map(joinClasses)
        
    dfthis = dfthis[~dfthis[target].isin(removeClasses)] # remove requested classes 
      
    y = np.ravel(dfthis[target])
    X = dfthis.drop(excludeFields + [target],axis=1)
    features = X.columns.values
    X = X.values


  
    min_max_scaler = preprocessing.MinMaxScaler()
    robust_scaler = preprocessing.RobustScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    X_robust = robust_scaler.fit_transform(X)
    quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
    X_quant = quantile_transformer.fit_transform(X)
    #quantile_transformer.quantiles_ 


    X_unscaled = X.copy()
    X = X_minmax.copy()
    #X = X_robust.copy()
    #X = X_quant.copy()
    
    return X, y, X_unscaled, features

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
def defClassifiers():

    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn import preprocessing

    
    import pickle
    min_max_scaler = preprocessing.MinMaxScaler()
    
    
    clfNames = ["Nearest Neighbors", "SVC", 
             #"RBF SVM",
           #  "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]
    
    clfNames=['knn','svc','dtc','rf','mlp','ada'] # nifty names 
    
    classifiersx = [ # with tuned parameters - redundant now
        KNeighborsClassifier(algorithm='brute',n_neighbors=5,p= 1, weights='distance'),
        SVC(kernel="linear",degree=6, C=0.025,probability=True),
        #SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(criterion='entropy',max_depth=10,
                               min_samples_leaf=5, min_samples_split=10),
       
        #RandomForestClassifier(max_depth=10,max_features=17, n_estimators=10),
         RandomForestClassifier(max_depth=10,n_estimators=10),
        MLPClassifier(alpha=1, 
                      activation='tanh',hidden_layer_sizes=(150, 97, 69, 29),
                      learning_rate='adaptive', solver='lbfgs'),
        AdaBoostClassifier(learning_rate=0.6,n_estimators=70),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    
    classifiers = [
        KNeighborsClassifier(),
        SVC(),
        #SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(),
       
        #RandomForestClassifier(max_depth=10,max_features=17, n_estimators=10),
         RandomForestClassifier(),
        MLPClassifier(),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
        ]
    return clfNames, classifiers
    

    
    
       
    #datasets = {'X':X,'minmax':X_minmax,'reduced':X_new}
    datasets ={} # only one hardcoded now 
    


results_mcc = {} 
results_cm = {}
results_cv ={} 
all_results = {}
all_series_results = {}
excludeFields={}

seriesDescription = {'s0':'No prediction of diagnosis using audio, untuned',
                      's1':'No prediction of diagnosis using audio, tuned',
                      's2':'real diagnosis added',
                      's6':'tuned rf, lowminmax',
                      's7':'tuned rf, lowminmax, include Audio',
                      's8':'all clf tuned: lowminmax:exclude Audio',
                      's9':'all clf tuned: lowminmax:include Audio',
                      's10':'all clf tuned: lowminmax:include Audioannd diagnosed',
                      's11':'all clf tuned:low var excl:classes 1,2;3,4 joined',
                      's12':'s11 + audio',
                      's13':'deduped dataset',    
                      'rs01':'rest dataset run 1',   
                      'rs02':'rest dataset run 2, -gravity, -maxEntropy',
                      'rs03':'rest dataset run 3, +gravity, -maxEntropy',
                      'rs04':'rest dataset run 4, +gravity,audioprob',
                   
                      'rs05':'rest dataset run 5, acel sigs ',
                      'rs06':'rest dataset run 6, acel sigs + jerk  ',
                        'rs07':'rest dataset run 6, device sigs + jerk  ',
  'rs08':'rest dataset run 8, 3 classes , device sigs + jerk, extratrees feature',
  'rs09':'rest dataset run 9, 5 classes , extratrees feature, favoured calsses ',
    'rs10':'rest dataset run 10, 5 classes, extratrees feature, more data',
    'rs11':'rest dataset run 11, 5 classes , extratrees feature,ordinal,  more data, tuned alg',
                      's3':'3&4 combined, -1 excluded',
                      's5':'new gait data and m sigs,low var and corr vars removed',
                      's4':'0,1 combined, 2,3,4 combined -1 on its own'}
seriesDesc = {}
seriesDescription['s3'] = '3&4 combined, -1 excluded'
seriesDescription['s4'] = '0,1 combined, 2,3,4 combined -1 on its own'

excludeFields['s0'] = ['healthCode','diagnosed','audioProb','updrs2_12']
excludeFields['s1'] = ['healthCode','diagnosed','audioProb','updrs2_12']
excludeFields['s2'] = ['healthCode','audioProb','audioProb','updrs2_12']
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
excludeFields['s14'] = ['healthCode','diagnosed','audioProb','updrs2_12']
excludeFields['rs01'] = ['healthCode','diagnosed','audioProb','updrs2_10']
excludeFields['rs02'] = ['healthCode','diagnosed','audioProb','updrs2_10',
             'maxEntropy']
excludeFields['rs03'] = ['healthCode','diagnosed','audioProb','updrs2_10']
excludeFields['rs04'] = ['healthCode','diagnosed','updrs2_10']
excludeFields['rs05'] = ['healthCode','diagnosed','audioProb','updrs2_10']
excludeFields['rs06'] = ['healthCode','diagnosed','audioProb','updrs2_10']

excludeFields['rs07'] = ['healthCode','diagnosed','audioProb','updrs2_10']
excludeFields['rs08'] = ['healthCode','diagnosed','audioProb','updrs2_10']
excludeFields['rs09'] = ['healthCode','diagnosed','audioProb','updrs2_10']
excludeFields['rs10'] = ['healthCode','diagnosed','audioProb','updrs2_10']
excludeFields['rs11'] = ['healthCode','diagnosed','audioProb','updrs2_10']
#%%
def reportdefinitions(series,voiceFeatures):
    global target
    global seriesDescription
    global lowvar
    global act 
    jc12134 ={-1:-1,0:0,1:1,2:1,3:2,4:2} # map 1,2 to 1, 3,4 to 2 
    jchilo ={-1:-1,0:0,1:0,2:1,3:1,4:1} # groups: -1, 0 (0,1) , 2 (2,3) 
    jchilo0 ={-1:-1,0:0,1:1,2:1,3:1,4:1} # groups: -1, 0 (0,1) , 2 (2,3) 
    
    joinClasses34 ={-1:-1,0:0,1:1,2:2,3:3,4:3} # map 3,4 to 3 
    
    # note these are features to be removed from evaluation, not added 
    collinear = ['nPulses', 'jitter_rap',
       'jitter_ppq5', 'shimmer_loc.1', 'shimmer_apq3']
    
    featuresMin = ['gender','healthCode','diagnosed','datetime','medPoint_0.0',
                     'medPoint_1.0','medPoint_2.0','medPoint_3.0','timediff',
                     'onset_year','diagnosis_year','updrs2_1',
                     'updrs2_10','updrs2_12']               
                     
    # remove colinnear from dataset 
    featuresMinCol = ['gender','healthCode','diagnosed','datetime','medPoint_0.0',
                     'medPoint_1.0','medPoint_2.0','medPoint_3.0','timediff',
                     'onset_year','diagnosis_year','updrs2_1',
                     'updrs2_10','updrs2_12'] + collinear                   
    
    # include onset rather than age                
    featuresMinOn_set = ['gender','healthCode','diagnosed','datetime','medPoint_0.0',
                     'medPoint_1.0','medPoint_2.0','medPoint_3.0','timediff',
                     'age','diagnosis_year','updrs2_1','updrs2_10','updrs2_12'] 
        
    featuresMax = ['healthCode','datetime','diagnosis_year','updrs2_1','updrs2_10',
                   'updrs2_12','timediff',] + collinear

    
    featuresMinPwP = ['age','gender','healthCode','diagnosed','datetime','medPoint_0.0',
                     'medPoint_1.0','medPoint_2.0','medPoint_3.0','timediff',
                     'onset_year','diagnosis_year','updrs2_1','updrs2_10',
                     'updrs2_12']  + collinear

    
    featuresMinMinPwP = featuresMinPwP + \
                   [voiceFeatures[i]  for i in np.arange(len(voiceFeatures)) if i not in [1,7,8,9,20] ] # top 5 voice features
    
    
    featuresMaxPwP = ['healthCode','diagnosed','datetime','medPoint_0.0', # age and gender
                     'medPoint_1.0','medPoint_2.0','medPoint_3.0','timediff',
                     'onset_year','diagnosis_year','updrs2_1','updrs2_10','updrs2_12']
    featuresMaxPwP1 = ['gender','healthCode','diagnosed','datetime','medPoint_0.0', # only age
                     'medPoint_1.0','medPoint_2.0','medPoint_3.0','timediff',
                     'onset_year','diagnosis_year','updrs2_1','updrs2_10','updrs2_12']
    featuresMaxPwP2 = ['age','healthCode','diagnosed','datetime','medPoint_0.0',
                     'medPoint_1.0','medPoint_2.0','medPoint_3.0','timediff',
                     'onset_year','diagnosis_year','updrs2_1','updrs2_10','updrs2_12'] + list(features[0:20]) 
    # set defaults: 

    
    jcl = joinClasses34

    #-------------------------
    if series== 'v01':
       
        seriesDescription[series] = 'Predict updrs2_1, Min feature set'
        excludeFields[series] = featuresMinCol
        remcl[series] = []
        lowvar=False
        target='updrs2_1'
        act='voice'
    #-------------------------
    if series== 'vcl1':
       
        seriesDescription[series] = 'predict updrs2_1,  minimum subset, excluding collinear'
        excludeFields[series] = featuresMinCol
        remcl[series] = []
        lowvar=False
        target='updrs2_1'
        act='voice'
    #-------------------------
    if series== 'v01lv':
       
        seriesDescription[series] = 'predict updrs2_1,  healthcodes time matched with walk and score'
        excludeFields[series] = ['healthCode','diagnosed','datetime','medPoint_0.0',
                     'medPoint_1.0','medPoint_2.0','medPoint_3.0',
                     'onset_year','diagnosis_year','timediff','updrs2_1','updrs2_10','updrs2_12']
        remcl[series] = []
        lowvar=0.01
        target='updrs2_1'
        act='voice'
    #-------------------------
    if series== 'v01ol':
       
        seriesDescription[series] = 'predict updrs2_1,  outliers removed (108 dropped)'
        excludeFields[series] = ['healthCode','diagnosed','datetime','medPoint_0.0',
                     'medPoint_1.0','medPoint_2.0','medPoint_3.0',
                     'onset_year','diagnosis_year','timediff','updrs2_1','updrs2_10','updrs2_12']
        remcl[series] = []
        lowvar=False
        target='updrs2_1'
        act='voice'
    #---------------
    if series== 'v01ol2':
       
        seriesDescription[series] = 'predict updrs2_1, robust scaler'
        excludeFields[series] = ['healthCode','diagnosed','datetime','medPoint_0.0',
                     'medPoint_1.0','medPoint_2.0','medPoint_3.0',
                     'onset_year','diagnosis_year','timediff','updrs2_1','updrs2_10','updrs2_12']
        remcl[series] = []
        lowvar=False
        target='updrs2_1'
        act='voice'
    #---------------

    elif series== 'v02':
   
        seriesDescription[series] = '''predict updrs2_1: Max Feature set'''
        excludeFields[series] = featuresMax
        remcl[series] = []
        lowvar=False
        target='updrs2_1'
        act='voice'
    #-------------------------
    elif series=='v03':
  
        seriesDescription[series] = 'predict updrs2_1: Removed -1: minimum subset features - f1_score'
        excludeFields[series] = featuresMinCol
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_1'
        act='voice'
    #-------------------------
    elif series=='vcl3':
  
        seriesDescription[series] = 'predict updrs2_1: Removed -1: minimum subset features; collinear excluded - f1_score'
        excludeFields[series] = featuresMinCol
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_1'
        act='voice'
    #-------------------------
    elif series=='v03mcc':

        seriesDescription[series] = 'predict updrs2_1: Removed -1: minimum subset features - mcc'
        excludeFields[series] = featuresMin
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_1'
        act='voice'
    #-------------------------
    elif series=='v03mad':
        
        seriesDescription[series] = 'predict updrs2_1: Removed -1: minimum subset features - mad'
        excludeFields[series] = featuresMin
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_1'
        act='voice'
    #-------------------------
    elif series=='v04':
    
        seriesDescription[series] = 'predict updrs2_1, medpoint, onset year and diagnosed included healthcodes time matched with walk and score'
        excludeFields[series] = featuresMax
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_1'
        act='voice'
    #-------------------------
    elif series=='v05':
        jcl = jchilo  # groups: -1, 0 (0,1) , 2 (2,3)
        seriesDescription[series] = 'predict updrs2_1, hi lo features Min '
        excludeFields[series] =  featuresMin 
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_1'
        act='voice'
         #-------------------------
    elif series=='v05.1':
        jcl = jchilo  # groups: -1, 0 (0,1) , 2 (2,3)
        seriesDescription[series] = 'predict updrs2_1, hi lo features Max '
        excludeFields[series] =  featuresMax 
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_1'
        act='voice'
         #-------------------------
    elif series=='v06':
        jcl = jchilo0 # map 0 ->0  12,3,4-> 1 
        seriesDescription[series] = 'predict updrs2_1, hi lo0 features min '
        excludeFields[series] = featuresMin
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_1'
        act='voice'
    #-------------------------
    
    elif series=='v07':
        jcl = jchilo0 # map  n-> n 3,4 -> 3  
        seriesDescription[series] = 'predict updrs2_1, hi lo0 features max '
        excludeFields[series] = featuresMax
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_1'

        act='voice'
#-------------------------
    elif series== 'v1001':
  
        seriesDescription[series] = 'predict updrs2_10,  healthcodes time matched with walk and score'
        excludeFields[series] = featuresMinCol
        remcl[series] = []
        lowvar=False
        target='updrs2_10'
        act='voice'
    #-------------------------
    elif series== 'v1002':
      
        seriesDescription[series] = '''predict updrs2_10: medpoint, onset year 
            and diagnosed included:suspect pd excluded:timediff > 1 hr 
            removed for medpoints > 0 exluded '''
        excludeFields[series] = featuresMax
        remcl[series] = []
        lowvar=False
        target='updrs2_10'
        act='voice'
    #-------------------------
    elif series=='v1003':
     
        seriesDescription[series] = 'predict updrs2_10: Removed -1: minimum subset features'
        excludeFields[series] = featuresMinCol
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_10'
        act='voice'
    #-------------------------
    elif series=='v1004':
        jcl = joinClasses34 # map  n-> n 3,4 -> 3  
        seriesDescription[series] = 'predict updrs2_10, medpoint, onset year and diagnosed included healthcodes time matched with walk and score'
        excludeFields[series] = featuresMax
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_10'
        act='voice'
    #-------------------------
    elif series=='v1005':
        jcl = jchilo  # groups: -1, 0 (0,1) , 2 (2,3)
        seriesDescription[series] = 'predict updrs2_10, hi lo features Min '
        excludeFields[series] =  featuresMinCol 
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_10'
        act='voice'
         #-------------------------
    elif series=='v1005.1':
        jcl = jchilo  # groups: -1, 0 (0,1) , 2 (2,3)
        seriesDescription[series] = 'predict updrs2_10, hi lo features: Max '
        excludeFields[series] =  featuresMax 
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_10'
        act='voice'
         #-------------------------
    elif series=='v1006':
        jcl = jchilo0 # map 0 ->0  12,3,4-> 1 
        seriesDescription[series] = 'predict updrs2_10, hi lo0 features min '
        excludeFields[series] = featuresMinCol
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_10'
        act='voice'
    #-------------------------
    
    elif series=='v1007':
        jcl = jchilo0 # map 0,1 -> 12,3,4-> 2 
        seriesDescription[series] = 'predict updrs2_10, hi lo0 features max '
        excludeFields[series] = featuresMax
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_10'
        act='voice'
    #-------------------------
    #-------------------------
    elif series== 'v1201':
  
        seriesDescription[series] = 'predict updrs2_10,  healthcodes time matched with walk and score'
        excludeFields[series] = featuresMinCol
        remcl[series] = []
        lowvar=False
        target='updrs2_12'
        act='voice'
    #-------------------------
    elif series== 'v1202':
      
        seriesDescription[series] = '''predict updrs2_12: medpoint, onset year 
            and diagnosed included:suspect pd excluded:timediff > 1 hr 
            removed for medpoints > 0 exluded '''
        excludeFields[series] = featuresMax
        remcl[series] = []
        lowvar=False
        target='updrs2_12'
        act='voice'
    #-------------------------
    elif series=='v1203':
     
        seriesDescription[series] = 'predict updrs2_12: Removed -1: minimum subset features'
        excludeFields[series] = featuresMinCol
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_12'
        act='voice'
    #-------------------------
    elif series=='v1204':
        jcl = joinClasses34 # map  n-> n 3,4 -> 3  
        seriesDescription[series] = 'predict updrs2_12, medpoint, onset year and diagnosed included '
        excludeFields[series] = featuresMax
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_12'
        act='voice'
    #-------------------------
    elif series=='v1205':
        jcl = jchilo  # groups: -1, 0 (0,1) , 2 (2,3)
        seriesDescription[series] = 'predict updrs2_12, hi lo features Min '
        excludeFields[series] =  featuresMinCol 
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_12'
        act='voice'
         #-------------------------
    elif series=='v1206':
        jcl = jchilo  # groups: -1, 0 (0,1) , 2 (2,3)
        seriesDescription[series] = 'predict updrs2_12, hi lo features: Max '
        excludeFields[series] =  featuresMax 
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_12'
        act='voice'
         #-------------------------
    elif series=='v1207':
        jcl = jchilo0 # map 0 ->0  12,3,4-> 1 
        seriesDescription[series] = 'predict updrs2_12, hi lo0 features min '
        excludeFields[series] = featuresMinCol
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_12'
        act='voice'
    #-------------------------
    
    elif series=='v1208':
        jcl = jchilo0 # map 0,1 -> 12,3,4-> 2 
        seriesDescription[series] = 'predict updrs2_12, hi lo0 features max '
        excludeFields[series] = featuresMax
        remcl[series] = [-1]
        lowvar=False
        target='updrs2_12'
        act='voice'
    #-------------------------
    #-------------------------
    elif series== 'vpd01':
    
        seriesDescription[series] = 'predict PD, - age and gender excluded '
        excludeFields[series] = featuresMinPwP
        remcl[series] = []
        lowvar=False
        target='diagnosed'
        act='voice'
    #-------------------------
    elif series== 'vpd03':
       
        seriesDescription[series] = 'predict PD, - age and gender included '
        excludeFields[series] = featuresMaxPwP
        remcl[series] = []
        lowvar=False
        target='diagnosed'
        act='voice'
    #-------------------------
    elif series== 'vpd02':
       
        seriesDescription[series] = 'predict PD, - only age  included '
        excludeFields[series] = featuresMaxPwP1
        remcl[series] = []
        lowvar=False
        target='diagnosed'
        act='voice'
    #-------------------------
    elif series== 'vpd04':
       
        seriesDescription[series] = 'predict PD, - only age , no signals  included '
        excludeFields[series] = featuresMaxPwP2
        remcl[series] = []
        lowvar=False
        target='diagnosed'
        act='voice'
    #-------------------------
    elif series== 'vpd21':
    
        seriesDescription[series] = 'predict PD, -include unknowns; age and gender excluded '
        excludeFields[series] = featuresMinPwP
        remcl[series] = []
        lowvar=False
        target='diagnosed'
        act='voice'
    #-------------------------
    elif series== 'vpd22':
       
        seriesDescription[series] = 'predict PD, -include unknowns;age and gender included '
        excludeFields[series] = featuresMaxPwP
        remcl[series] = []
        lowvar=False
        target='diagnosed'
        act='voice'
    #-------------------------
    elif series== 'vpd23':
       
        seriesDescription[series] = 'predict PD, - include unknowns;only age  included '
        excludeFields[series] = featuresMaxPwP1
        remcl[series] = []
        lowvar=False
        target='diagnosed'
        act='voice'
    #-------------------------
    elif series== 'vpd24':
       
        seriesDescription[series] = 'predict PD, - include unknowns;only age , no signals  included '
        excludeFields[series] = featuresMaxPwP2
        remcl[series] = []
        lowvar=False
        target='diagnosed'
        act='voice'
    #-------------------------
    
    return {'jc': jcl,'desc': seriesDescription[series],'excl':excludeFields[series],
                'remcl':remcl[series],'lowvar':lowvar,'target':target,'act':act}

def testCVpredict(clf,X,y):
    # with clf, predict scores using crossvalpredict 
    
    name= 'rf'
    params = [elem[1] for elem in [(prf1[name])] if elem[0]==series ]      
    print('Processing :',name, 'Tuned params: ',params[0]) 
    
    clf = classifiers[3]        
    clf.set_params(**params[0])
            
    from sklearn.model_selection import cross_val_predict
    
    y_pred = cross_val_predict(clf, X_hold, y_hold, cv=20)
    f1_score(y_hold, y_pred) 
    return

ff = getSeriesResults('v01','19aug')
#%%
def getSeriesResults(series, suff =''):
    ff =  pickle.load(open('all_results_series_pickle_' + suff + '_' + series,'rb'))
    return ff
#%%
class stats1(object):
       def __init__(self, clfs, series, stats):  
               self.clfs = clfs
               self.series = series
               self.seriesStats = stats
    
#%%
 
def reportResults(seriesList,resultsSuf,fn,subtitle):
   # only works on final results so far 
    statsDict = {}  
    
    suffix = resultsSuf
    
    for s in seriesList:
        results = getSeriesResults(s,suffix) 
        
        serFinal = s + '_Final' # only works on finals at the moment 
        clfs = list(results[serFinal].keys())
        
        stats = results[serFinal]
        
        statObj = stats1(clfs,s,stats) # leave series name without suffix 
        statsDict[s] = statObj

   # type(statObj.seriesStats)
    repTitle = 'Series ' + str(seriesList)
    
    reports.sampleRep(statsDict,repTitle,fn,subtitle)
    
    return
#%%
#import reports
import reports
imp.reload(reports)

seriesList21=['v01','v02','v03','v04','v05','v05.1','v06','v07']
subtitlev21 = 'Predicting MDS-UPDRS 2.1 from voice data'


seriesList210=['v1001','v1002','v1003','v1004','v1005','v1005.1','v1006','v1007']
subtitlev210 = 'Predicting MDS-UPDRS 2.10 from voice data'

seriesList212=['v1201','v1202','v1203','v1204','v1205','v1206','v1207','v1208']
subtitlev212 = 'Predicting MDS-UPDRS 2.12 from voice data'

seriesListol = ['v01ol','v01ol2','v01']
subtitleol = 'Predicting MDS-UPDRS 2.1 , outliers removed | robust scaler'

seriesListpdplus = ['vpd21','vpd22','vpd23']
subtitlepdplus = 'Predicting diagnosed, inc. doubtfuls, quantatative '

seriesListpd = ['vpd01','vpd02','vpd03']
subtitlepd = 'Predicting diagnosed vs undiagnosed '
seriesList = seriesListpd
subtitle = subtitlepd
fn = 'v_series_pd_'

reportResults(seriesList,resultsSuf,fn,subtitle)
rdf = {}

completeSeriesList = seriesList21 + seriesList210 + seriesList212 + seriesListpd
 

for s in completeSeriesList:
    rdf[s] = reportdefinitions(s,[])
    

reports.printSeriesRegister(rdf)    

printSKFResults('v12',suffix='14aug')
ser = 'v12'; suffix = '14aug' 
   resultstt = getSeriesResults(ser,suffix) 
#%%   
    
def printSKFResults(series,suffix='',others=False):
    ''' printing f1 score and MAD scores for each fold '''
   
    #suffix = resultsSuf

    results = getSeriesResults(series,suffix) 
 

    try:   
        print('Results for series ',series,' : ',seriesDescription[series])
    except KeyError:
        print('Results for series ',series,' : no description available')

        # print the suffixed scores, averaged 
    seriesNumeric = [k for k in results.keys() if k.startswith(series) & \
                 k[-1].isdigit()]
    if not others:
        for clfname in results[seriesNumeric[0]].keys(): # all clfs same in each suffix
            print('--------------- SKF :   ',clfname,'  --------')
        
            for stat in ['MAD1','MAD2','f1_macro']:
                dd = np.array([results[a][clfname][stat] for a in seriesNumeric])
                avg = sum([results[a][clfname][stat] for a in seriesNumeric])/len(seriesNumeric)
                std = np.std(dd)
                print('{0:s}: mean: {1:3.3f} std {2:3.3f}'.format(stat,avg,std))
                
    seriesList = [k for k in results.keys() if k.startswith(series) ]        
    otherSeries = set(seriesList) - set(seriesNumeric)
    
    for series in otherSeries:
        for clfname in results[series].keys(): 
            print('--- Series ',series, ' :   ',clfname,'  --------')
            for stat in ['MAD1','MAD2','f1_macro']:
                print(stat,': ',results[series][clfname][stat])
    return

def print1Series():  
    series = 'v01'
    suffix='19aug'
    results = getSeriesResults(series,suffix) 
    reportdefinitions(series)
    series = 'v01_Final'
  
    printSKFResults(series,suffix=suffix,others=True)
           
def printSeriesResults(series,classes,results,cmIndex='CM1',subprint=False,clf=False):

    seriesNumeric = [k for k in results.keys() if k.startswith(series) & \
                 k[-1].isdigit()]
    seriesList = [k for k in results.keys() if k.startswith(series) ]  
    desc = seriesDescription[series]
    if subprint: seriesList=subprint 
    for subseries in seriesList:
        print('Series: ',subseries)
        for clfname in results[subseries].keys():
            
            if clf: 
                if clfname not in clf:continue
                   
        #print(all_results[name][1])
        #name='knn';series='s1'
        #classes = [-1,0,1,2,3]
            try:
               print(results[subseries][clfname][cmIndex])
               
               plot_confusion_matrix(results[subseries][clfname][cmIndex],
                         title=clfname + ' ' + desc,classes=classes,
                         normalize=False)
            except KeyError:
               print('no results for ',clfname)
    return       

# ---------------------------------------------------------------------------       
def printSeriesF1macro(series):
    names = ['timeNow', 'logloss', 'cm',
                              'cmf','cmo','mad_rate','mad_rate_f','mad_rate_o']
    
    
    print(seriesDescription[series])
    for name in all_series_results:
    #print(all_results[name][1])
        try:
            for k,v in all_series_results[name][series].items():
                print(name,': ',k,' : ',v)
        except:
            print('no results for name :',name )            

#%%
def getOrdClass(clf,X_train, X_test,y_train,y_test,para):
    '''A Simple Approach to Ordinal Classification
    Eibe Frank and Mark Hall '''
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y_traint = le.fit_transform(y_train)
    y_testt = le.fit_transform(y_test) # -1 - 4 -> 0-5
    numClasses = len(le.classes_)
    probs = [0]*numClasses
    fprobs = [0]*numClasses
    #clf.set_params(**para) #reapply tuned parameters - required ? 
    for cl in range(numClasses): 
        # create binary of (classes <= value, classes > val)
        if cl == numClasses-1:
            fprobs[cl] = probs[cl-1]
            continue
        # create 2 classes - 1 if > cl, else 0 
        y_traintt = [1 if i > cl else 0 for i in y_traint]
        y_testtt = [1 if i > cl else 0 for i in y_testt]
        # run classiffier for this split, getting prob 
        clf.fit(X_train, y_traintt)
        #y_predtt = clf.predict(X_test)
        y_probatt = clf.predict_proba(X_test)
        print('cl is ',cl)
        probs[cl]= y_probatt[:,[1]] # prob class > cl
        # prob for this class cl = (prob > previous class) - (prob > this class) 
        # i.e. probs[cl-1] - probs[cl] 
        #edge cases: cl = 0 ... 1-probs[cl]
        #            cl = max ...probs[max-1]  
        # note that the last class prob is just prob > last class -1 is 
        # handled at the top of the loop
        if cl == 0: fprobs[cl] = 1-probs[cl]
        else: fprobs[cl] = probs[cl-1] - probs[cl]
        
    winner = np.argmax(np.array(fprobs),axis=0)
    winner = winner.flatten()
    return le.inverse_transform(winner)     

def testOrdClass():
    X_train = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    y_train = [-1,0,1,2]
    X_test =  X_train.copy()
    y_test = y_train.copy()
    
#%% -----------------------------------------------------------------------------
def mad_rate(y_ground, y_pred):
     return sum(abs(y_pred-y_ground))/len(y_ground) #

 
#def logSeriesResults(name,seriesSuf,cm,cmo,logloss,f1_macro,mad_rate,mad_rate_o):
def logSeriesResults(clfname,seriesSuf,series,trainDist,testDist,cm1,cm2,logloss,
                     f1_macro,mad1,mad2,trainshape,testshape,params,lowvar,
                     features,importancies):
                             
    global all_series_results2
    
    timeNow = datetime.datetime.now().isoformat(' ', 'seconds') 
    
    seriesEntry = {'time':timeNow,'CM1':cm1,'CM2':cm2,'logloss':logloss,
            'f1_macro':f1_macro,'MAD1':mad1,'MAD2':mad2,'trainshape':trainshape,
            'testshape':testshape,'params':params,'lowvar':lowvar,
            'features':np.array(features),'importancies':importancies}
     
    if seriesSuf in all_series_results2.keys():
        all_series_results2[seriesSuf].update({clfname:seriesEntry})
 
    else: #new algorithm - add to results dict 
        all_series_results2[seriesSuf] = {clfname:seriesEntry}
    
#    if name in all_series_results.keys():
#        all_series_results[name].update({seriesSuf:{
#            'time':timeNow,'CM1':cm,'CM2':cmo,'logloss':logloss,
#            'f1_macro':f1_macro,'MAD1':mad_rate,'MAD2':mad_rate_o}})
# 
#    else: #new algorithm - add to results dict 
#        all_series_results[name] = {seriesSuf:{
#            'time':timeNow,'CM1':cm,'CM2':cmo,'logloss':logloss,
#            'f1_macro':f1_macro, 'MAD1':mad_rate,'MAD2':mad_rate_o}}
    return

#%%  
def calcScores(df,X,y,series,joinClasses,target,clfNames,classifiers,prf,
               removeClasses,clfName=0,varT=0,tuned=False,lowvar=False,
               useHoldout=True,holdout=[0,0,0,0],seriesFeatures=[],
               resultsSuf=''): 
    ''' clfName - to limit scores to one clf  
        varT - low variance threshold - now replaced by lowvar itself 
        lowvar - False or threshold 
        prf - tuned parameters from gridsearch '''
    from sklearn.model_selection import StratifiedKFold
    #create datasets based on exclude fields 
    # X_train, X_test, y_train, y_test, df_train, \
        #    df_test = splitByHealthCode(df,excludeFields,0.7)
    #
    from sklearn.metrics import precision_recall_fscore_support as scoreAll
    
#    X, y, X_unscaled, features = splitFeatureDF(df,excludeFields[series],
#                                                joinClasses,removeClasses,
#                                                target)
   #todo remove collinear (see below - manual at the moment)
    if lowvar:
        Xlow, selectedLowVar, variances = removeLowVariance(X,lowvar) # 0,015 for walk reduces to 30 - got by experimentation
        print('low variance removed, threshold: ',lowvar,' shape: ',Xlow.shape)
        X = Xlow.copy()
    #Xlow, importances = drExtraTrees(X_unscaled, y,features,False) # use extratrees to get most important feats 
    #print('number of features femoved with var < ',lowvar,X[0].size - Xlow[0].size)
   # print(pd.Series(y).value_counts())
        # compare each classifier on same split 
   # top10Features = sorted(zip(importances,features),reverse=True)[:10]    
    print('calcScores: series ',series,'. Number of features: ',X.shape[1])   
   
    
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,
                                                        stratify=y)
    X_trainbk = X_train.copy(); X_testbk= X_test.copy() # save for svc (which reduces the dimensions)
    trainDist = pd.Series(y_train).value_counts()
    testDist = pd.Series(y_test).value_counts()
    print('class distribution train:',trainDist)
    print('class distribution test:',testDist)
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    unqClasses = np.unique(y_train)
    
    for name, clf in zip(clfNames, classifiers):
        if (name == 'svc') & (~lowvar): # svc performance poor with many features
            
            X_train, importancies = drExtraTrees(X_train,y_train,seriesFeatures) # all above median
            X_test, importancies = drExtraTrees(X_test,y_test,seriesFeatures)
        else: 
            X_train = X_trainbk;X_test = X_testbk # restore ful ldimension x_train/test 
            
        if clfName: # i.e. one classifier has been passed in for scoring 
            if name != clfName: continue
      
        #if name in ['dtc','knn','svc']:continue
        # set estimator parameters from gridsearchCV if prf1
        
        if tuned: # apply gridsearch parameters 
            params = [elem[1] for elem in [(prf1[name])] if elem[0]==series ]      
            print('Processing :',name, 'Tuned params: ',params[0]) 
            clf.set_params(**params[0]) # dict inside  lists
        else:
            print('Processing :',name, ' - untuned ') 
            params = [{}]
                        
        if name == 'svc': clf.set_params(probability=True)
        counter = 0
        # if passed multiple clfs, run skf on train dataset , else
        # for one classifier, predict test results 
# --- one classifier 
        if clfName: 
            if useHoldout: 
                X_train, X_test, y_train, y_test = holdout 
            seriesSuf  = series + '_Final'
            clf.fit(X_train, y_train) # train on train set 
            y_pred= clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)        
            y_ord = getOrdClass(clf,X_train, X_test,y_train,y_test,
                                    params[0])
            f1_macro=f1_score(y_test, y_pred,average='macro')
  
            f1_macro_ord =f1_score(y_test, y_ord, average='macro')
            mccscore = mcc(y_test,y_pred) 
            mad_rate = sum(abs(y_pred-y_test))/len(y_test) # mean absolute deviation            
            mad_rate_o = sum(abs(y_ord-y_test))/len(y_test)
            cm = confusion_matrix(y_test,y_pred)
            cmo = confusion_matrix(y_test,y_ord)
            logloss = log_loss(y_test,y_proba)
            try:
                featureImportance = clf.feature_importances_
            except:   
                featureImportance = -1
            #logSeriesResults(name,seriesSuf,cm,cmo,logloss,f1_macro,mad_rate,mad_rate_o)
            logSeriesResults(name,seriesSuf,series,trainDist,testDist,cm1=cm,
                              cm2=cmo,logloss=logloss,f1_macro=f1_macro,
                              mad1=mad_rate,mad2=mad_rate_o,
                              trainshape=X_train.shape,testshape=X_test.shape,
                              params=params[0],lowvar=lowvar,features=features,
                              importancies=featureImportance
                              )
    
        else: # i've been passed many clfs to assess 
        # only use training dataset to train and assess - train 
        # is split into new 3 train and test sets  ... 
            
            skf = StratifiedKFold(n_splits=5)
            
            for train_idx, test_idx in skf.split(X_train, y_train):
                counter += 1
                seriesSuf  = series + '_'  + str(counter) 
                X_train_f = X_train[train_idx]
                y_train_f = y_train[train_idx]
                
                clf.fit(X_train_f, y_train_f)
                try:
                    featureImportance = clf.feature_importances_
                except:   
                    featureImportance = -1
                
                X_test_f = X_train[test_idx]
                y_test_f = y_train[test_idx] 
                print('skf train shape: ',X_train_f.shape,'skf test shape: ',X_test_f.shape) 
               # score = clf.score(X_test_f, y_test_f)
                y_pred= clf.predict(X_test_f)
                y_proba = clf.predict_proba(X_test_f)
                len(y_proba)
            
                y_ord = getOrdClass(clf,X_train_f, X_test_f,y_train_f,y_test_f,
                                    params[0])
                y_ord.shape

                precision, recall, fscore, support = scoreAll(y_test_f, y_pred,
                                                          average='macro')

                f1_macro=f1_score(y_test_f, y_pred,average='macro')
            #pdb.set_trace()
       
                f1_macro_ord =f1_score(y_test_f, y_ord, average='macro')
                mad_rate = sum(abs(y_pred-y_test_f))/len(y_test) # mean absolute deviation
          
                mad_rate_o = sum(abs(y_ord-y_test_f))/len(y_test)
            # normailsed mad rate ? at the moment favouring populous classes 
            #df_test.shape
            #y_pred_hc, y_test_hc = majorityScore(y_pred,df_test)
                cm = confusion_matrix(y_test_f,y_pred)
      
                cmo = confusion_matrix(y_test_f,y_ord)
                try:
                    logloss = log_loss(y_test_f,y_proba)
                except ValueError:
                    logloss = -1
        
               # cm_hc = None # only for multiple records per healthCode 
    
       
           # cm_hc = confusion_matrix(y_test_hc,y_pred_hc)
        #len(y_pred)
        #results_mcc[name] = cross_val_score(clf, item, y, cv=5,scoring=mc_scorer)
                timeNow = datetime.datetime.now().isoformat(' ', 'seconds') 
# --- log results                                
                logSeriesResults(name,seriesSuf,series,trainDist,testDist,cm1=cm,
                              cm2=cmo,logloss=logloss,f1_macro=f1_macro,
                              mad1=mad_rate,mad2=mad_rate_o,
                              trainshape=X_train_f.shape,testshape=X_test_f.shape,
                              params=params[0],lowvar=lowvar,features=features,
                              importancies=featureImportance
                              )
          
        pickle.dump(all_series_results2, 
        open('all_results_series_pickle_' + resultsSuf + '_' + series,"wb")) 
        timeNow = datetime.datetime.now().isoformat(' ', 'seconds')
        print('Completed :',name, 'time:', timeNow) 
        
#%%        
def calcScores2(df,X,y,series,joinClasses,target,clfNames,classifiers,prf,
               removeClasses,clfName=0,varT=0,tuned=False,lowvar=False,
               useHoldout=True,holdout=[0,0,0,0],seriesFeatures=[],
               resultsSuf=''): 
    ''' clfName - to limit scores to one clf  
        varT - low variance threshold - now replaced by lowvar itself 
        lowvar - False or threshold 
        prf - tuned parameters from gridsearch '''
    from sklearn.model_selection import StratifiedKFold
    #create datasets based on exclude fields 
    # X_train, X_test, y_train, y_test, df_train, \
        #    df_test = splitByHealthCode(df,excludeFields,0.7)
    #
    from sklearn.metrics import precision_recall_fscore_support as scoreAll
    
#    X, y, X_unscaled, features = splitFeatureDF(df,excludeFields[series],
#                                                joinClasses,removeClasses,
#                                                target)
   #todo remove collinear (see below - manual at the moment)
    if lowvar:
        Xlow, selectedLowVar, variances = removeLowVariance(X,lowvar) # 0,015 for walk reduces to 30 - got by experimentation
        print('low variance removed, threshold: ',lowvar,' shape: ',Xlow.shape)
        X = Xlow.copy()
    #Xlow, importances = drExtraTrees(X_unscaled, y,features,False) # use extratrees to get most important feats 
    #print('number of features femoved with var < ',lowvar,X[0].size - Xlow[0].size)
   # print(pd.Series(y).value_counts())
        # compare each classifier on same split 
   # top10Features = sorted(zip(importances,features),reverse=True)[:10]    
    print('calcScores: series ',series,'. Number of features: ',X.shape[1])   
   
    
    unqClasses = np.unique(y_train)
    
    for name, clf in zip(clfNames, classifiers):
       
        if clfName: # i.e. one classifier has been passed in for scoring 
            if name != clfName: continue
      
        #if name in ['dtc','knn','svc']:continue
        # set estimator parameters from gridsearchCV if prf1
        
        if tuned: # apply gridsearch parameters 
            params = [elem[1] for elem in [(prf1[name])] if elem[0]==series ]      
            print('Processing :',name, 'Tuned params: ',params[0]) 
            clf.set_params(**params[0]) # dict inside  lists
        else:
            print('Processing :',name, ' - untuned ') 
            params = [{}]
                        
        if name == 'svc': clf.set_params(probability=True)
    
        # if passed multiple clfs, run skf on train dataset , else
        # for one classifier, predict test results 

    
         # i've been passed many clfs to assess 
        # only use training dataset to train and assess - train 
        # is split into new 3 train and test sets  ... 
        counter = 0    
        skf = StratifiedKFold(n_splits=5)
        f1_macro = {}
        f1_macro_ord = {}
        mad_rate = {}
        mad_rate_o = {}
        cm1 = {}
        cm2={}
        logloss = {}
        for train_idx, test_idx in skf.split(X, y): # /* USE WHOLE DATASET */ 
            counter += 1
            seriesSuf  = series + '_'  + str(counter) 
            X_train_f = X[train_idx]
            y_train_f = y[train_idx]
            
            clf.fit(X_train_f, y_train_f)
            try:
                featureImportance = clf.feature_importances_
            except:   
                featureImportance = -1
            
            X_test_f = X[test_idx]
            y_test_f = y[test_idx] 
            trainDist = pd.Series(y_train_f).value_counts()
            testDist = pd.Series(y_test_f).value_counts()
            print('class distribution train:',trainDist)
            print('class distribution test:',testDist)
            print('skf train shape: ',X_train_f.shape,'skf test shape: ',X_test_f.shape) 
           # score = clf.score(X_test_f, y_test_f)
            y_pred= clf.predict(X_test_f)
            y_proba = clf.predict_proba(X_test_f)
            len(y_proba)
        
            y_ord = getOrdClass(clf,X_train_f, X_test_f,y_train_f,y_test_f,
                                params[0])
            y_ord.shape

            precision, recall, fscore, support = scoreAll(y_test_f, y_pred,
                                                      average='macro')

            f1_macro[counter] =f1_score(y_test_f, y_pred,average='macro')
        #pdb.set_trace()
   
            f1_macro_ord[counter] =f1_score(y_test_f, y_ord, average='macro') # ordclass predictions
            mad_rate[counter] = sum(abs(y_pred-y_test_f))/len(y_test_f) # mean absolute deviation
      
            mad_rate_o[counter] = sum(abs(y_ord-y_test_f))/len(y_test_f)  # ordclass predictions
        # normailsed mad rate ? at the moment favouring populous classes 
        #df_test.shape
        #y_pred_hc, y_test_hc = majorityScore(y_pred,df_test)
            cm1[counter] = confusion_matrix(y_test_f,y_pred)
  
            cm2[counter] = confusion_matrix(y_test_f,y_ord)
            try:
                logloss[counter] = log_loss(y_test_f,y_proba)
            except ValueError:
                 logloss[counter] = -1
    
           # cm_hc = None # only for multiple records per healthCode 

   
       # cm_hc = confusion_matrix(y_test_hc,y_pred_hc)
    #len(y_pred)
    #results_mcc[name] = cross_val_score(clf, item, y, cv=5,scoring=mc_scorer)
            timeNow = datetime.datetime.now().isoformat(' ', 'seconds') 
# --- log results                                
            logSeriesResults(name,seriesSuf,series,trainDist,testDist,cm1=cm1[counter],
                          cm2=cm2[counter],logloss=logloss[counter],
                          f1_macro=f1_macro[counter],mad1=mad_rate[counter],
                          mad2=mad_rate_o[counter],
                          trainshape=X_train_f.shape,testshape=X_test_f.shape,
                          params=params[0],lowvar=lowvar,features=features,
                          importancies=featureImportance
                          )
    
          
          
        # log final results   
        seriesSuf = series + '_Final'
        # calcultate mean and std devation of all runs 
    
        f1_mean = np.mean([x for x in f1_macro.values()])
        f1_std =  np.std([x for x in f1_macro.values()])
        mad1_mean = np.mean([x for x in mad_rate.values()])
        mad1_std =  np.std([x for x in mad_rate.values()])
        mad2_mean = np.mean([x for x in mad_rate_o.values()])
        mad2_std =  np.std([x for x in mad_rate_o.values()])
        cm1_total = sum(cm1.values()) # sum confusion matrices 
        cm2_total = sum(cm2.values()) # sum confusion matrices
        logloss_mean = np.mean([x for x in logloss.values()])
        
        # note using params, train and test shape to store sd's 
        
        logSeriesResults(name,seriesSuf,series,trainDist,testDist,cm1=cm1_total,
              cm2=cm2_total,logloss=logloss_mean,f1_macro=f1_mean,
              mad1=mad1_mean,mad2=mad2_mean,
              trainshape=mad1_std,testshape=mad2_std,
              params=f1_std,lowvar=lowvar,features=features,
              importancies=featureImportance
              )  
        
        pickle.dump(all_series_results2, 
        open('all_results_series_pickle_' + resultsSuf + '_' + series,"wb")) 
        timeNow = datetime.datetime.now().isoformat(' ', 'seconds')
        print('Completed :',name, 'time:', timeNow) 
    return 
#%% dimensionality reduction feature selection - low variance and collinearity 
# low variance 
def removeLowVariance(X,thresh):
    ''' apply variance threshold of x , return transformed, mask and variances  '''
    
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=thresh)
    var = sel.fit(X)
    selectedLowVar = sel.get_support()
    #features[~selectedLowVar]
    Xlow = sel.fit_transform(X)
    return Xlow, selectedLowVar, var.variances_

def removeCollinear(X):
    '''remove collinear vars - manual at the moemnt  '''
    Xlow = X.copy()
    Xlow = X
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
    
    #22-26  low eigenvalues 
    len(v[22])
    
    v[:,17].tolist()
    for f in [22,23,24,25,26]:
        print(f,[i for i,v in enumerate(v[:,f]) if abs(v) >  0.5])
    
    [i for i,v in enumerate(v[:,23]) if abs(v) >  0.5]
    
    # it seems 5 and 7 and 17 and 20 are collinear - what are these ? 
    featuresArray = np.array(features)
    featuresArray[[5,6,12,13,14,15,16,17,18]]
    featuresArray[[17]]
    from pandas.plotting import scatter_matrix
    dfscat = pd.DataFrame(X)
    scatter_matrix(dfscat,figsize=[20,20],marker='x')
    dfscat = pd.DataFrame(X[:,[5,6,12,13,14,15,17,18]])
    scatter_matrix(dfscat,figsize=[20,20],marker='x')
    cc1 = np.corrcoef(X[:,[5,6,12,13,14,15,17,18]],)
    
    x = X[:,22]
    y = X[:,23]
    colors = (0,0,0)
    area = np.pi*3
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    featuresArray[[selectedLowVar]].tolist()
    # drop 7 and 17 
    Xlow.shape
    tt = np.delete(Xlow,[7,17],1)
    tt.shape
    Xlow = tt.copy()
# Xlow has all low variance and crrelated variables removed 
#%% dimentionality reduction extratrees 
def drExtraTrees(X,y,features,plot=False,etThresh='median'):
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    model = SelectFromModel(clf,prefit=True,threshold=etThresh)
    X_new = model.transform(X)
    X_new.shape 
    X.shape        
    
    importances = clf.feature_importances_
   
    
    # graphically 
    if plot:
        impList = list(zip(features,importances  ))  
        sortedImp = sorted(impList, key=lambda imp: imp[1],reverse=True)  
        print(sortedImp)
       
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
    
    return X_new, importances        
#%% tune algorithms  
# tune promising models : adaboost % : )
def gridSearch(X, y, features,series,hcGroups,lowvar):
    ''' hcgroups passed through for group kfolds ''' 
    
    tuningResults = {}    
    from sklearn.model_selection import GridSearchCV
    
    
    classNames, classifiers  = getclfs('voice')
    # from sci-kit learn  http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    C_range = np.logspace(-2, 9, 6)
    gamma_range = np.logspace(-9, 2, 6) 
    
    if lowvar: 
        Xlow, selectedLowVar, variances = removeLowVariance(X,lowvar) # 0,015 for walk reduces to 30 - got by experimentation
        print('low variance removed, threshold: ',lowvar,' shape: ',Xlow.shape)
        X = Xlow.copy()
        
    max_features = np.arange(5,X.shape[1],3)
    
    parameters = [{'n_neighbors':[3,5,7],'weights':['uniform','distance'], #knn
                     'algorithm':['brute','ball_tree','kd_tree'],'p':[1,2]},
                     [  {'kernel': ['rbf'], 'gamma': gamma_range,   #svc
                         'C': C_range,'class_weight':['balanced']},
                       # {'kernel': ['linear'], 'C': [1, 10, 100, 1000],'class_weight':['balanced']},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['poly'], 'C': [1, 10, 100, 1000],'class_weight':['balanced']}
                     ],
                  {"criterion": ["gini", "entropy"], #dtc
                  "min_samples_split": [2, 10, 20],
                  "max_depth": [None, 2, 5, 10],
                  "min_samples_leaf": [1, 5, 10],
                  "max_leaf_nodes": [None, 5, 10, 20],
                  'class_weight':['balanced']
                  },
                  {'max_features':max_features,
                   'max_depth':[2,4,6,8,10], #rf
                  'n_estimators':[10,15,17,20],'class_weight':['balanced']},
                  {'hidden_layer_sizes':[(15,10),(25,10,3),(70,20,5),(35,35,35),
                                         (20,11,11,5)], #mlp
                                         'activation':['tanh','relu'],
                                         'solver':['lbfgs','adam'],
                                         'learning_rate':['constant','adaptive'],
                                         'learning_rate_init':[0.05, 0.01, 0.005, 0.001]},
                  {'n_estimators':[30,50,70,100],'learning_rate':[0.5,0.6,0.7,0.8,1]}, #adaboost
                  {'class_weight':['balanced']}, # extratrees 
                  {'strategy':['stratified','most_frequent']} # dummy
                  #,{'GuassianNB - only priors'},
                  #{'QDA - no hyperparameters'},
                   ]

    

    ftwo_scorer = make_scorer(fbeta_score, beta=2)
    f1_scorer = make_scorer(f1_score)
    mcc_scorer = make_scorer(mcc)
    precision_macro = make_scorer(precision_score, average='macro')
    mad_scorer  = make_scorer(mad_rate, greater_is_better=False)
    
    # gkf=GroupKFold(n_splits=5) future 
    #classif = 2
#    noclasses = np.unique(y).size
#    if noclasses == 2:scorer = mcc_scorer
#    else: scorer = 'f1_macro'
    for classifier, parameter,clName in zip(classifiers,parameters,classNames):
           
        
        clf = GridSearchCV(classifier, parameter,scoring ='f1_macro')
        #clf.fit(X, y,groups=hcGroups)
        clf.fit(X, y)
        
        clf.cv_results_
        print("Grid scores on training set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        tuningResults[clName] = [series,clf.cv_results_['params'][clf.best_index_]]
        pickle.dump(tuningResults,open('tuningResults_series_'+ series,'wb'))
#%% Ensemble learn from best 3 
def ensembleClassify(df,series,joinClasses,target,clfNames,classifiers,prf1,
               removeClasses,clfName=0,lowvar=0):
    ''' aim: take the best classifiers, their gridsearch parameters 
    and use to get the best guess (highest probability ) from each classifier '''
    from sklearn.ensemble import VotingClassifier
    X, y, X_unscaled, features = splitFeatureDF(df,excludeFields[series],
                                                joinClasses,removeClasses,
                                                target)
    # knn, rf and mlp
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
   
    clfs={#'dtc':DecisionTreeClassifier(),
           'knn':KNeighborsClassifier(),
          'rf':RandomForestClassifier(),
          #'mlp':MLPClassifier(),
          'svc':SVC(kernel="linear", C=0.025,probability=True)}
#    for clfName, clf in clfs.items():
#        params = [elem[1] for elem in [(prf1[clfName])] if elem[0]=='v05' ] 
#        #print(params[0])
#        #print(type(params[0]))
#        clf.set_params(**params[0])
    b = list(zip(clfs.keys(),clfs.values()))
    #get parameter {"criterion": ["gini", "entropy"],
    parameters = [
            {'knn__n_neighbors':[3,5,7],'knn__weights':['uniform','distance'], #knn
                     'knn__algorithm':['brute','ball_tree','kd_tree'],'knn__p':[1,2]},
            #  {"dtc__criterion": ["gini", "entropy"],
                 # "dtc__min_samples_split": [2, 10, 20],
                 # "dtc__max_depth": [None, 2, 5, 10],
                # "dtc__min_samples_leaf": [1, 5, 10],
                 # "dtc__max_leaf_nodes": [None, 5, 10, 20],
                 # },
                  {'rf__max_features':[8,10,12,14],'rf__max_depth':[2,4,6,8,10],
                  'rf__n_estimators':[10,15,17,20]},
                   #[
                           {'svc__kernel': ['rbf'], 'svc__gamma': [1e-3, 1e-4],
                         'svc__C': [0.1,1,10,100]},
                      #  {'svc__kernel': ['linear'], 'svc__C': [1, 10, 100, 1000]},
                    #    {'svc__kernel': ['poly'], 'svc__C': [1, 10, 100, 1000]}
                     #],
                   
                #  {        'mlp__hidden_layer_sizes':[(15,10),(25,10,3),(70,20,5),(35,35,35)],
                           #              'mlp__activation':['tanh','relu'],
                           #              'mlp__solver':['lbfgs','adam'],
                           #              'mlp__max_iter':[500],
                           #              'mlp__learning_rate':['constant','adaptive'],
                           #              'mlp__learning_rate_init':[0.05,  0.005]}
                   ]
    from sklearn.model_selection import GridSearchCV
    v_clf = VotingClassifier(estimators=b,voting='soft')
    grid = GridSearchCV(estimator=v_clf, param_grid=parameters, cv=5)
    grid = grid.fit(X_train,y_train)
   # v_clf.fit()
    y_pred=grid.predict(X_test)
    MAD1 = mad_rate(y_test,y_pred) 
    print('MAD1: ', MAD1)
    f1_macro=f1_score(y_test, y_pred,average='macro')
    print('f1_macro',f1_macro)
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    precision, recall, fscore, support = scoreAll(y_test, y_pred)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))    
#%%
def getclfs(act):
    
    classifiers = [
        KNeighborsClassifier(5),
        SVC(kernel="linear", C=0.025),
        #SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=10,max_features=17, n_estimators=10),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        ExtraTreesClassifier(),
        DummyClassifier()
        
       # ,GaussianNB(),
       # QuadraticDiscriminantAnalysis()
        ]
    classNames=['knn','svc','dtc','rf','mlp','ada','ext','dum'
                #,nb','qda'
                ]
    
    return classNames,classifiers
            
#%% Load walk data, calculate scores 
# make sure scores (from the getScores function that reads the scores dataset)
#  are loaded before this ! 
df, droppedRows = loadWalkData(rest=True,runNo=1)
joinClasses ={-1:-1,0:0,1:1,2:1,3:2,4:2} # map 1,2 to 1, 3,4 to 2 
joinClasses ={-1:0,0:0,1:1,2:1,3:2,4:2} # map 1,2 to 1, 3,4 to 2 
jc34 ={-1:-1,0:0,1:1,2:2,3:3,4:3} # map 4 to 3 
joinClasses = False
jc = {} # dict of join classes 
seriesDescription = {}
remcl = {} 
lowvar=0.005
df.head()



X, y, X_unscaled, features = splitFeatureDF(df,excludeFields[series],
                                        jc[series],remcl[series],target)

X_train, X_hold, y_train, y_hold = \
        train_test_split(X, y, test_size=.25, random_state=251,
                         stratify=y)

y_a = y_hold
plt.hist(y_a, bins=np.arange(y_a.min(), y_a.max()+1))

pd.Series(y).value_counts()
#Xlow, selectedLowVar = removeLowVariance(X,lowvar)
#clf = KNeighborsClassifier(algorithm='brute',n_neighbors=5,p= 1, weights='distance')


clfNames,classifiers = getclfs('walk')


calcScores(df,X_train,y_train,series,jc[series],target,clfNames,classifiers,
           prf1,remcl[series],clfName=0,varT=0.0001,tuned=False,lowvar=False,
            useHoldout=False,holdout=[X_train,X_hold,y_train,y_hold]) 

classes = np.unique(y)
printSKFResults(series)

Xlow, importances = drExtraTrees(X, y,features) # cut dimensionality in half 

# series picks up from previous calcscores run 
# note gridsearch completed on x_train, y_train above, hold out set not seen by gridsearch 
# holdout set used in final evaluation 

gridSearch(X_train, y_train, features, series,lowvar=lowvar)

prf1 = pickle.load(open('tuningResults_series_'+series,'rb')) # load tuning parameters

clfNames,classifiers = getclfs('walk')

# Calculate tuned scores using SKF 
calcScores(df,X_train,y_train,series,jc[series],target,clfNames,classifiers,
           prf1,remcl[series],clfName=0,varT=varT,tuned=True,lowvar=lowvar,
           useHoldout=False,holdout=[X_train,X_hold,y_train,y_hold])

printSKFResults(series)
printSeriesResults(series,classes,all_series_results2)


# with selected, tuned algorith, predict test scores - results are put in 'xnn_Final'

calcScores(df,X_hold,y_hold,series,jc[series],target,clfNames,classifiers,
           prf1,remcl[series],clfName='dtc',varT=0.001,tuned=True,lowvar=lowvar,
            useHoldout=True,holdout=[X_train,X_hold,y_train,y_hold]) 
# --- here

printSKFResults(series + '_Final')

printSeriesResults(series + '_Final',classes,all_series_results2)



# select best performing classifiers from previous step 
# not you have to hard code the classifiers in the function at the moment 
ensembleClassify(df,series,jc[series],target,clfNames,classifiers,
           prf1,remcl[series],0,varT) 

#%%
def removeNonPwP(df):
    df2 = df.copy()
    df2.columns
    possibles = (df2['diagnosed'] == False) & \
             ((df2['diagnosis_year']>1900) | (df2['onset_year']>1900))
    removed = sum(possibles)         
    df2 = df2[~possibles]
        
    return df2, removed

def removeTimeDiff(df,hours):
    ''' remove th
    ose records with time gap > 1.5 hours, diagnosis = true, 
    medpoint > 0 '''
    df2 = df.copy()

    secs = hours*3600
    toRemove = (df2['diagnosed'] == True) & \
             (df2['medPoint_0.0'] == 0) & (df2['timediff']>secs*1000)
    removed = sum(toRemove)         

    df2 =  df2[~toRemove]        
     
    return df2, removed

def addTotalCol(df):
    df['updrstot'] = df.updrs2_1+df.updrs2_10+df.updrs2_12
    df['updrstot'] = df['updrstot'].map(lambda x: -1 if x == -3 else x)
    return df 

#ggg = getSeriesResults('v03',suff='19aug')

#%% load voice data, calculate scores 
df, droppedRows = loadVoiceData(5)
# exclude all with onset year or diagnosed year but not diagnosed 
df2  =df.copy()
df2.shape

df2, removed = removeNonPwP(df2)
df2.shape
df2.diagnosed.value_counts()

df2, removed = removeTimeDiff(df2,1.5)
df2.shape
df2.diagnosed.value_counts()

df2 = df2[df2.updrs2_10!=-2]
df2.isnull().sum()
voiceFeatures = df2.columns[0:23].values


# manual removal of outliers ... redundant if using robustscaler, which I now am 
outliers = getOutlier(df,4)
print('No rows with outliers :',len(outliers))
df2.updrs2_1.value_counts()
df4 = df2.drop(outliers)
df4.updrs2_1.value_counts()

df2.shape
jc = {} # dict of join classes 
seriesDescription = {}
remcl = {} 
varT=0.005
lowvar=False
act='walk'
target=''



jc12134 ={-1:-1,0:0,1:1,2:1,3:2,4:2} # map 1,2 to 1, 3,4 to 2 
jchilo ={-1:-1,0:0,1:0,2:1,3:1,4:1} # groups: -1, 0 (0,1) , 2 (2,3) 
jchilo0 ={-1:-1,0:0,1:1,2:1,3:1,4:1} # groups: -1, 0 (0,1) , 2 (2,3) 
    
joinClasses34 ={-1:-1,0:0,1:1,2:2,3:3,4:3} # map 3,4 to 3 
    
featuresMin = ['healthCode','diagnosed','datetime','medPoint_0.0',
                     'medPoint_1.0','medPoint_2.0','medPoint_3.0','timediff',
                     'onset_year','diagnosis_year','updrs2_1','updrs2_10','updrs2_12']
featuresMax = ['healthCode','datetime','updrs2_1','updrs2_10','updrs2_12','timediff']              

#%% testing reportdef as class 
from copy import deepcopy
class reportDef(object):
    
       def __init__(self, series,jc,desc,excF,remcl,lowvar,target,act):  
             
               self.series = series
               self.jc = jc
               self.desc = desc
               self.excF = excF
               self.remcl = remcl
               self.lowvar = lowvar
               self.target = target
               self.act = act 
               
       def print(self):
           print(self.act)
# convert report def to class - underconstruction 

rv01 = reportDef(series = 'v01',
               jc = joinClasses34,
               desc = 'predict updrs2_1,  healthcodes time matched with walk and score',
               excF = featuresMin,
               remcl = [],
               lowvar = False,
               target = 'updrs2_1',
               act = 'voice' ) 
           
rv08 = deepcopy(rv01)
rv08.series = 'v08'
rv08.target = 'updrs2_10'
rv08.desc = 'predict updrs2_10,  healthcodes time matched with walk and score'

# --------------------------


#%% Begin voice data algorithm evaluation 
# gridsearch parameters - use entire data set 
series='v02'
resultsSuf='29aug2'
rdef = reportdefinitions(series,voiceFeatures)

target

# following is temporary fix to onset year for old data 
#df2['onset_year'] = df2['onset_year'].apply(lambda x : 0 if x==0 else 2015-x)
#df2 = pickle.load(open('.//reports//savedf2','rb'))
# remove features, join classes as required by series 
X, y, X_unscaled, features = splitFeatureDF(df2,excludeFields[series],
                                            rdef['jc'],remcl[series],target)
ff = pd.DataFrame(X) ## to get into variable inspector 
# need to add groups here too ... 
Xlow, importances = drExtraTrees(X, y,features,True) # cut dimensionality in half 
Xlow.shape

X_train, X_hold, y_train, y_hold = \
        train_test_split(X, y, test_size=.25, random_state=251,
                         stratify=y)
    
Xlowv, selectedLowVar, variances_ = removeLowVariance(X_unscaled,lowvar)
type(variances_)

a = list(zip(features,list(variances_)))
variances = sorted(a,key=lambda a:a[1]) # can use this to see low variance features
# this is for whole dataframe - need to get into each split for kgroups  
hcGroups = syn1.getHCGroups(df2)
X_train.shape
X[0]

pd.Series(y_hold).value_counts()

pd.Series(y_train).value_counts()

#gridSearch(X_train, y_train, features,series,hcGroups,lowvar=lowvar)
# why not use whole dataset for gridsearh ? 
gridSearch(X, y, features,series,hcGroups,lowvar=lowvar)
 
len(y_train)        
prf1 = pickle.load(open('tuningResults_series_'+series,'rb'))


seriesFeatures[series] = [f for f in list(features) if f not in excludeFields[series]]

#import copy
#asr_bu = copy.deepcopy(all_series_results2)  

pd.Series(y).value_counts()
X_train.shape
   
clfNames,classifiers = getclfs(act)
varT=0
# model evaluation - k-fold validation on training set only 

#calcScores('df',X_train,y_train,series,rdef['jc'],target,clfNames,classifiers,
#           prf1,remcl[series],0,varT=0.001,tuned=True,lowvar=lowvar,useHoldout=False,
#           holdout=[X_train,X_hold,y_train,y_hold],
#           seriesFeatures=seriesFeatures[series],resultsSuf=resultsSuf)
##series='v07'
#printSKFResults(series,resultsSuf)
#printSKFResults('vcl3',resultsSuf)
#classes = np.unique(y_train) 
#printSeriesResults(series,classes,all_series_results2)
## --- voice final run
##all_series_results2['v06_Final'].pop('svc')
#
## need to add x-fold validation, get p and 95% CI - not possible ? 
#top3 = ['knn','rf','ext']
top3 = ['mlp','svc','ext','rf','knn','ada','dtc','dum']
#top3=clfNames
#for clf in top3: # predict using holdout set  
#    calcScores('df',X_hold,y_hold,series, rdef['jc'],target,clfNames,classifiers,
#               prf1,remcl[series],clfName=clf,varT=0.001,tuned=True,lowvar=lowvar,
#                useHoldout=True,holdout=[X_train,X_hold,y_train,y_hold],
#                seriesFeatures=seriesFeatures[series],resultsSuf=resultsSuf) 
# 
# predict using entire set and kfold stratified x-validation    
for clf in top3: 
    calcScores2('df',X,y,series, rdef['jc'],target,clfNames,classifiers,
               prf1,remcl[series],clfName=clf,varT=0.001,tuned=True,lowvar=lowvar,
                useHoldout=True,holdout=[X_train,X_hold,y_train,y_hold],
                seriesFeatures=seriesFeatures[series],resultsSuf=resultsSuf)     
# use cross val score on final model
#series='v1205.1'
printSKFResults(series,resultsSuf)

printSKFResults(series,resultsSuf,others=True)
# printSKFResults('vpd01',resultsSuf,others=True)
# series='v1006';classes=[0,1];clfselect='rf'

clfselect='rf'
classes = [0,1]
series = 'vpd03'
printSeriesResults(series,classes,all_series_results2,cmIndex='CM1',
                   subprint=[series + '_Final'],clf=clfselect)

cvscores = {}
skf = StratifiedKFold(n_splits=3)
# ------ final model evaluation 


# take hold out set and test predictions of the final models 
for model in top3: 
    clf = [e[1] for e in list(zip(clfNames, classifiers)) if e[0] ==model][0]
    
    paraDict = prf1[model][1]
    type(paraDict)
    clf.set_params(**paraDict)
    t = cross_val_score(clf,X_hold,y_hold, cv=skf,scoring='f1_macro')
    print("f1_macro of ",model,": %0.2f (+/- %0.2f)" % (t.mean(), t.std() * 2))




ensembleClassify(df2,series,rdef['jc'],target,clfNames,classifiers,
           prf1,remcl[series],0,varT) 

series = 'v01'
unqClasses = [-1,0,1,2,3,4]
printSeriesResults(series,unqClasses,all_series_results2,5) 
printSeriesF1macro(series)


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
#%% experiment with feature selection 



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
# 
#random forest , SVC, knn 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

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

#%% Drive gridsearch    

df, droppedRows = loadWalkData(rest=False,runNo=1)
joinClasses ={-1:-1,0:0,1:1,2:1,3:2,4:2} # map 1,2 to 1, 3,4 to 2 
joinClasses ={-1:0,0:0,1:1,2:1,3:2,4:2} # map 1,2 to 1, 3,4 to 2 
joinClasses = False
target='updrs2_12' 
gridexFields = ['healthCode','diagnosed','audioProb',target]
removeClasses = []
X, y, X_unscaled, features = splitFeatureDF(df,gridexFields,joinClasses,
                                removeClasses,target)
        #todo remove collinear (see below - manual at the moment)

Xlow, importances = drExtraTrees(X, y,features) # cut dimensionality in half 
Xlow.shape
# series picks up from previous calcscores run 
gridSearch(Xlow, y, features, series)

prf1 = pickle.load(open('tuningResults_series_'+series,'rb'))
#%%
y_pred= c.predict(X_test)
len(y_pred)
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt



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

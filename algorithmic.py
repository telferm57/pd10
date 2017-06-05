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
imp.reload(sp)

sp.featureList

# load into pd for clean-up
df = pd.read_csv(cfg.WRITE_PATH + '/Features.csv2')
df.columns=sp.featureList
df.shape
df.head(100)
df.isnull().sum()


len(df)
len(df.dropna())
df.dropna(inplace=True)
df['gender']=df['gender'].map({'Male':1,'Female':0})
df['jitter_x']=df['jitter_x'].apply(lambda x:float(x))
df['jitter_x'].iloc[13853]
df['gender'].hist()
df['updrs'].hist()
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
    
    
yy = df.iloc[:,-1:].values
yy = np.ravel(yy)
yy[1]
z = list(map(lambda x:int(x),yy))
y = z

X = df.iloc[:,:-2].values

#%%
# 
#random forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm 

from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support

c = RandomForestClassifier()
c= svm.SVC()
b = DummyClassifier() # generates predictions by respecting the training set's class distribution

results = []
baselines = []

for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    c.fit(X_train, y_train)
    b.fit(X_train, y_train)
    res = c.score(X_test, y_test)
    bas = b.score(X_test, y_test)
    print('Loop', i, res, bas)
    results.append(res)
    baselines.append(bas)

print( '\nBaseline', np.mean(baselines), np.std(baselines))
print( 'Random Forest', np.mean(results), np.std(results))
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
y_pred= c.predict(X_test)
len(y_pred)
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)
#%%
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=[-1,0,1,2,3,4],normalize=True,
                      title='Confusion matrix, with normalization')
plt.figure()
plot_confusion_matrix(cm, classes=[-1,0,1,2,3,4],normalize=False,
                      title='Confusion matrix, with normalization')
plt.show()
#%%

squares = list(map(lambda x: x**2, range(10)))
print(squares)

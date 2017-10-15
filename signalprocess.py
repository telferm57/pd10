# -*- coding: utf-8 -*-
"""
Created on Tue May 23 21:44:35 2017

@author: dad
"""

# read in file 
#WRITE_PATH = 'F:\\DS\\diss\\data\\'
import pandas as pd
import numpy as np
import cfg
cfg.WRITE_PATH
file1 = "accel_walking_return-test-1.csv"
file2 = 'accel_walking_rest.json.items5388181.csv5'
file3 = 'accel_walking_return.json.items5388132.csv5'
import os
cwd = os.getcwd()
walkout=pd.read_json(cfg.WRITE_PATH+'/testfiles/' +file1)
walkout.columns
walkout=walkout[['timestamp','x','y','z']]
walkrest=pd.read_csv(cfg.WRITE_PATH+file2,header=0)
walkrest.columns
walkrest=walkrest[['timestamp','x','y','z']]

#thanks to nlathia pydata_2016

#%%
import matplotlib
import matplotlib.pyplot as plt

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)
    
def plot_activity(activity):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, activity['timestamp'], activity['x'], 'x Axis')
    plot_axis(ax1, activity['timestamp'], activity['y'], 'y Axis')
    plot_axis(ax2, activity['timestamp'], activity['z'], 'z Axis')
    plt.subplots_adjust(hspace=0.2)
    plt.show()
    
plot_activity(walkout)
#plot_activity(walkrest)
#%% calc magnitude of vector 

import math

def magnitude(activity):
    x2 = activity['x']*activity['x']
    y2 = activity['y']*activity['y']
    z2 = activity['z']*activity['z' ]

    m2 = x2 + y2 + z2
    m = m2.apply(lambda x: math.sqrt(x))
    return m

walkout['magnitude'] = magnitude(walkout) 
#walkrest['magnitude'] = magnitude(walkrest) 

def plot_magnitudes(activities, titles):
    activities =[walkout]
    titles = ['walkout']
    fig, axs = plt.subplots(nrows=len(activities), figsize=(15, 15), squeeze=False)
    axs.shape
    for i in range(0, len(activities)):
        print('ededededed',i)
        i=0
        plot_axis(axs[i], activities[i]['timestamp'], activities[i]['magnitude'], titles[i])
    plt.subplots_adjust(hspace=0.2)
    plt.show()

plot_magnitudes([walkout],['walkout'])
#%%
def windows(df, size=100):
    start = 0
    while start < df.count():
        yield start, start + size
        start += (size // 2)

fig, ax = plt.subplots(nrows=1, figsize=(15, 3))
plot_axis(ax, walkout['timestamp'], walkout['magnitude'], 'walkout')

for (start, end) in windows(walkout['timestamp']):
    ax.axvline(walkout['timestamp'][start], color='r')
    
plt.show()
walkout.head()
#%%
from scipy.stats import skew, kurtosis
from statsmodels.tsa import stattools

import math

def jitter(axis, start, end):
    j = float(0)
    for i in range(start, min(end, axis.count())):
        if start != 0:
            j += abs(axis[i] - axis[i-1])
    return j / (end-start)


def mean_crossing_rate(axis, start, end):
    cr = 0
    m = axis.mean()
    for i in range(start, min(end, axis.count())):
        if start != 0:
            p = axis[i-1] > m
            c = axis[i] > m
            if p != c:
                cr += 1
    return float(cr) / (end-start-1)

def window_summary(axis, start, end):
    print(start,end)
    acf = stattools.acf(axis[start:end])
    acv = stattools.acovf(axis[start:end])
    sqd_error = (axis[start:end] - axis[start:end].mean()) ** 2
    return [
        jitter(axis, start, end),     
        mean_crossing_rate(axis, start, end),
        axis[start:end].mean(),
        axis[start:end].std(),
        axis[start:end].var(),
        axis[start:end].min(),
        axis[start:end].max(),
        acf.mean(), # mean auto correlation
        acf.std(), # standard deviation auto correlation
        acv.mean(), # mean auto covariance
        acv.std(), # standard deviation auto covariance
        skew(axis[start:end]),
        kurtosis(axis[start:end]),
        math.sqrt(sqd_error.mean())
    ]

def features(activity):
    featureList = ['jitter_x',
                   'jitter_y',
                   'jitter_z',
                   'jitter_m',
                   'mcr_x',
                   'mcr_y',
                   'mcr_z',
                   'mcr_m',
                   'mean_x',
                   'mean_y',
                   'mean_z',
                   'mean_m',
                   'std_x','std_y','std_z',
                   'std_m',
                   'var_x',
                   'var_y','var_z','var_m',
                   'min_x','min_y','min_z','min_m',
                   'max_x','max_y','max_z','max_m',
                   'acfm_x','acfm_x','acfm_x','acfm_x',
                   'acfm_s','acfm_s','acfm_s','acfm_s',
                   'acvm_x','acvm_x','acvm_x','acvm_x',
                   'acvm_s','acvm_s','acvm_s','acvm_s',
                   'skew_x','skew_y','skew_z','skew_m',
                   'kurt_x','kurt_y','kurt_z','kurt_m',
                   'sqrt_x','sqrt_y','sqrt_z','sqrt_m']
    
    
    for (start, end) in windows(activity['timestamp']):
        features = []
        for axis in ['x', 'y', 'z', 'magnitude']:
            features += window_summary(activity[axis], start, end)
        yield features
        #import csv
#%%
#activities = [walkout, walkrest]
activities = [walkout]
import csv


with open(cfg.WRITE_PATH + '../Data/Features.csv', 'w') as out:
    rows = csv.writer(out)
    for i in range(0, len(activities)):
        for f in features(activities[i]):
            rows.writerow([i] + f)

#os.listdir()
#os.chdir('./mjfoxdata')
#%%
dataset = np.loadtxt( cfg.WRITE_PATH + '/Data/Features.csv', delimiter=",")
X = dataset[:, 1:]
y = dataset[:, 0]
#%%
# 
#random forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split

c = RandomForestClassifier()
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
#mucking 
for i in range(0, len(activities)):
    print(i)


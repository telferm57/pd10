# -*- coding: utf-8 -*-
"""
Created on Mon May 29 19:08:59 2017

@author: dad
"""
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
                   'sqrt_x','sqrt_y','sqrt_z','sqrt_m',
                   'age','gender','profDiag','updrs']

def magnitude(activity):
    x2 = activity['x']*activity['x']
    y2 = activity['y']*activity['y']
    z2 = activity['z']*activity['z' ]

    m2 = x2 + y2 + z2
    m = m2.apply(lambda x: math.sqrt(x))
    return m

def windows(df, size=100):
    start = 0
    while start < df.count()-50:
        yield start, start + size
        start += (size // 2)
        
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
    try:
        tt = len(axis)
    except:
        print('exception on length of x')
        print('type of x: ',type(axis))
        print(axis)
    #print('type of x: ',type(axis))
    #print(start,end)
    
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
    for (start, end) in windows(activity['timestamp']):
        features = []
        for axis in ['x', 'y', 'z', 'magnitude']:
            features += window_summary(activity[axis], start, end)
        yield features
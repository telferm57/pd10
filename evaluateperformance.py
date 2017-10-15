# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:34:23 2017

@author: dad
"""

df.columns
trStem = 'tuningResults_series_'
series = 'v11'
tuningres = pickle.load(open(trStem +  series,'rb'))
arStem = 'all_results_series_pickle_14aug_'
algres = pickle.load(open(arStem +  series,'rb'))
# print importancies 


imp = algres[series]['ada']['importancies']
for k,v in algres.items(): print(k)


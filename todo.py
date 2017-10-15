# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:40:25 2017

@author: dad
"""

''' to do list 

A. DONE upgrade synapse client to 1.7.1  pip install --upgrade synapseclient
B. move to picking individual files based on demographics rather than dowloading the whole lot
C. add years since onset as a feature
D. get rid of globals ?
E. rewrite bits as class ? File handler ? 
F. getscores and demographics - remove need to call separately if not files not loaded
-- AT THE MOMENT YOU HAVE TO CALL THEM WITH 0 TO LOAD THE DATASET 
PRIOR TO using the function to query the data 
G get updrs score closest to date of file it is being added to 
G.2 at the moment get datafile downloads just the first file 
H. Praat script - add column headers to output file 
I.  turn mechanism for balancing the PwP and nPwpP into a function ?
at the moment it is hard coded to keep the updrs scores of the motor section balanced between 0,1 on one side, 
2+ on the other (simply because I began with the walking data ) ... but perhaps would be better 
to use disgnosed or not ? balance it 50-50 with diagnosed or not 
J add col headers to accel file ?
K add report at the end of each run with summary stats :
    no. helathcodes processed, pwp, pnwp, pwp who have no vpice score
    pwp who have no walk data, pwop who have no voice score 
L Change report to class
'''
 
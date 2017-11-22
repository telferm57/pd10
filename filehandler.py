# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:08:04 2017

@author: telferm

file handling processses

1. create a registry of files, 

2. check their completeness (i.e. does it contain the correct variables  ) 

3. check validity (Nan, equal length, values in range)

Files expected in following format: 

    
    matlab files containing sensor and time data 
    
"""
import pandas as pd
import numpy as np
import scipy.io as spio
from scipy import stats
from datetime import datetime


def createRegistry(basedir):
    ''' construct dict of files for each subdirectory , 
    keyed on subdirectory, giving int ID to each file ''' 
    
    #basedir = 'C:/Users/telferm/projects/vital/Batch1/'
    import os
    tt = os.listdir(basedir)
    reg = {}
    for ff in tt:
        reg[ff] = os.listdir(basedir + ff)    
    return reg

def matlabDate(intime):
    ''' returns python datetime from matlab timestamp ''' 
    
    if intime.dtype != 'float64':
        print('I need float64s ')
        return -1
    import datetime
    days1900 = 719529 # days 1/1/1900 -> 1/1/1970
    psx = (intime - days1900)*3600.0*24.0
    t = datetime.datetime.utcfromtimestamp(psx)
    return t, psx

def checkComplete(basedir,reg):
    ''' check files contain the correct variables :
        realTime : timestamps
        press : pressure readings 
        indxDn : index for pressure readings 
        acc1 : accelerometer readings 
            
        and that all except press are same length and have same time start and end 
        TODO check for ranges
        '''
    varsrequired = ['press','indxDn','acc1','realTime']
    filestats = {}
    report = {} 
    for dirs,files in reg.items():
       # if dirs == 'vah002':break
       report[dirs] = {}
       for file in files:
           
           print(file)
           matdata = spio.loadmat(basedir+dirs+'/' + file)
           md = matdata['myData']
           lens = []
           for var in varsrequired:
               
               check = md[var][0,0]  
               if var != 'acc1': check = np.ravel(md[var][0,0])
               filestats[file + '_' + var] = len(check)
               
               if var == 'realTime':
                   starttime,_ = matlabDate(check[0])
                   starttime = starttime.isoformat()[:-7]
                   endtime,_ = matlabDate(check[-1])
                   endtime = endtime.isoformat()[:-7]
                   filestats[file + '_time'] = [dirs,starttime, endtime] 
                   report[dirs][file]=[starttime[:10], starttime[-8:],
                          endtime[-8:],len(check)]
    # create output table subject:dir:date:datert:end:samples 
    #TODO check all lengths (time and/or samples ) are the same 
    return report 

def getAnnotations(subject,basedir):
    
    fn = subject + '.txt'
    subdir = 'annotations/'
    annotations = pd.read_table(basedir+subdir+fn,header=None,
                                index_col=False,
                                names=['section','it','start','end','duration',
                                       'notes'])
    return annotations

def getMatlabvars(mydata):
        
       # for var in varsrequired:
        ts = np.ndarray.flatten(mydata['realTime'][0,0])
        press = mydata['press'][0,0][:,1] 
        indxDn = np.ravel(mydata['indxDn'][0,0]) -1    
 
        axyz = mydata['acc1'][0,0]
        df = pd.DataFrame(axyz,columns=['x','y','z'])
         
        # assemble the pressure data .. 
        pressar = np.ones(len(indxDn))
        indxDn[-1]
        
        for i,j in enumerate(indxDn):
            pressar[i] = press[j]
           # if i == 5:break
        df['press'] = pressar
        
        def matlabDate(intime):
            ''' returns python datetime from matlab timestamp ''' 
            import datetime
            days1900 = 719529 # days 1/1/1900 -> 1/1/1970
            psx = (intime - days1900)*3600.0*24.0
            t = datetime.datetime.utcfromtimestamp(psx)
            return t, psx
       
      
        # import 
        ts.transpose().shape
        psxar = [x[1] for x in (map(matlabDate,ts))]
        datetimear = [x[0] for x in (map(matlabDate,ts))]
        #TODO - this is slow - should do in one command 
        
        df['timestamp'] = psxar
        df['datetime'] = datetimear
        
        return df 

def getMatlabfile(fn,basedir,subject):
    ''' inputs matlab file, outputs dataframe with acc, press and timestamps in datetime 
    object and posix value'''
    matdata = spio.loadmat(basedir+subject+'/' + fn)
    md = matdata['myData']
    mddf = getMatlabvars(md)
    
    return mddf
    #print(fn)
    
    

def createFilereport(basedir,subdir):   
    #basedir = 'C:/Users/telferm/projects/vital/Batch1/'
    reg = createRegistry(basedir+subdir)
    report = checkComplete(basedir+subdir,reg)
    rdf  = pd.DataFrame([])
    
    for subject, data in report.items():
        for file,stats in data.items():
            if file.split(sep='_')[1] == 'v':
                adl=0
            else: adl=1
            ser = pd.Series({'subject':subject,
                             'date':stats[0],
                             'start':stats[1],
                             'end':stats[2],
                             'samples':stats[3],
                             'fn':file,
                             'adl':adl})
            #print(ser)
            rdf = rdf.append(ser,ignore_index=True)
            
    return report, rdf
            
# calculate sma for vah002 

from vitalsensor import getSMA,windows,getActivityClasses, getVectors
import pickle

def multifileSMA(reg,basedir,subjects='all',windowsz=1,gravity=9.81):
    ''' for files in rdf, perhaps limited to subject , calculate the SMA values per window size (Secs)
    output to disk 
        '''
    fileprefix = './data/SMA'
    smafiles = {}     
    vectorfiles = {}
    for subject,files in reg.items():
        if subjects != 'all':
            if subject not in subjects:
                continue
        print('processing subject: ',subject)
            
       #  getsma for this one 
        for fn in files:
        
            print('processing file: ',fn)
            sensdat = getMatlabfile(fn,basedir,subject) 
            print(type(sensdat))
            #data = df.copy()
            
            ar, arb, argr, ars, vacc, Vhacc = getVectors(sensdat,
                                                         start=0,
                                                         end=-1,
                                                         gvalue=gravity)
            vects = {k:v for k,v in zip(['ar', 'arb', 'argr', 'ar',' vacc', 'Vhacc'],
                                        [ar, arb, argr, ars, vacc, Vhacc])}
            
            filevectors = './data/vectors'

            smaar = getSMA(arb)
            filepath = fileprefix + '_' + subject + '_' + fn
            pickle.dump(smaar, open(filepath,'wb'))
            smafiles.setdefault(subject,[]).append(filepath)
            
            filepath = filevectors + '_' + subject + '_' + fn
            pickle.dump(vects, open(filepath,'wb'))
            vectorfiles.setdefault(subject,[]).append(filepath)
            
            
    return smafiles, vectorfiles


def multifileActclasses(smafiles,gravity=9.81):
    '''construct acivity arrays for each input file, which are sma values over a
    one minute window 
    output as one nested dict of subject -> file -> sma 
    #TODO vary window size ''' 
    fileprefix = './data/SMA' #TODO make global or whatever  
    actfiles = {}
    for subject,files in smafiles.items():
        filepath = './data/SMA_vah002_vah002_h_28_00A096236B0F.mat'
        for filepath in files:
            #filepath = fileprefix + '_' + subject + '_' + fn #TODO make global or whatever
            sma = pickle.load(open(filepath,'rb'))
            activities = getActivityClasses(sma,g=gravity)
            fn = ''.join(filepath.split(sep='_')[2:])
            actfiles.setdefault(subject,{}).update({fn:activities})
            
    return actfiles
#%% Pressure Data
def getPressureData(fn,nrows=1e4,skiprows=0):
    import pandas as pd
    path='C:/Users/telferm/python/projects/walkingsignals/walkingsignals/data/smm/pilot/' 
    smm = pd.read_csv(path + fn,header=None,skiprows=skiprows,nrows=nrows)
    return smm 
            


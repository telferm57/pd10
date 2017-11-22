# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:31:20 2017

@author: telferm

basic workflow handling for sensor files. Once the basics are identified, 
look at implementing in a python pipline like luigi or airflow 

input to this process is a registry of files that comprise a number of days of
sensor data , perhaps with gaps of time and mutiple files per day . 
we then create a registry of time periods where the device is being worn 
('on body'). These are the time periods that will be analysed

Do we create a new set of files that are only on body activity ? I think not

We could create one file per day though , that would simplify. 
then run 'plugins' to perform each analysis.

plugins can be sensor processors, aggregators or visualisers 

format of sensor plugin: 
    
    input: 
        sensor file with timestamps and sensor data 
        OR output of previous plugin , such as window values for SMA 
        
        
        descriptor of sensor data - must map the following : 
            Gyro axes: gyrox, gyroy, gyroz 
            pressure: Press ,mmHG 
            Accelerometer: x,y,z m/sec , gravity included 
        
        prerequisties: e.g. Gait analysis requires walk segments and activity 
            levels 
            
    output: 
        report file which is suitable for visualiser andor aggregator plugins 
        
        
    

"""

''' sample first plugin - SMA  '''
def pluginSMA(data,datadescriptor,prereq,windowsz):
    ''' anadoning for now 13/11 11:40 - taking too much time away ! ''' 
    
def pluginActivityLevel(data,datadescriptor,prereq): # obv will be class 
    ''' activity level needs only accelerometer data and SMA (signal magnitude
    area) to generate a classification of activity in timeslots 
        data: accelerometer data 
        datadescriptor : this is actually SMA for a bunch of windows that relates 
        to a partiular file 
        
    '''
    # check data matches expected data descriptor
    # check prereq available 
    
    
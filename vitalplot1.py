# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 05:40:15 2017

@author: telferm
"""


#%% plotting helpers
import matplotlib.pyplot as plt
from matplotlib import patches 

import pandas as pd
from datetime import timedelta
import matplotlib.dates as mdates

def addGrid():
    ax = plt.gca()
    ax.minorticks_on()
    ax.grid(which='both')
    ax.legend()

#ax.set_xticks(tst.values)
def pltAnnotations(ax,annotations):
    annottimes=mdates.date2num(annotations.synched.tolist())
    ax2 = ax.twiny()
    ax.set_xticks(annottimes)
    ax.set_xticklabels(annottimes,rotation = 45)
    ax2.set_xticks(annottimes)
    ax2.set_xticklabels(annotations.notes2.values,rotation=70)
    ax.set_xlim(annottimes[0],annottimes[-1])
    #ax2.set_xlim(tst[0],tst[-1])
    ax2.set_xlim(annottimes[0],annottimes[-1])
    ax.set_ylim(-25,25)
    ax2.set_ylim(-25,25)
    timeFmt = mdates.DateFormatter('%M:%S')
    ax.xaxis.set_major_formatter(timeFmt)
    return ax2.plot(),ax.plot()
    
def plotVector(v,epoch,window):
    ''' create x axis for vector v , starting at start of epoch 
    assuming each element is window seconds from last. place each
    value in the middle of the window. window defined in seconds '''
    # so for SMA we have window of 1 second 
    window = 1
    td = timedelta(seconds = window)
    base = datetime(2000, 1, 1)
    timearray = np.arange(epoch[0]+np.timedelta64(seconds=window/2), 
                          epoch[1],np.timedelta64(window,'s'), 
                          dtype='datetime64')
    # convert to datetime 
    
    return pd.to_datetime(timearray)




def plotWalkTimes(ax, walklog,epoch,rate=50, param_dict={}):
    from datetime import timedelta
    import matplotlib.dates as mdates 
    ''' given a walklog object, that records all walks over a predfined sensor
    epoch, and a set of axes, draw the blocks of time on the axes 
    corresponding to walks > 2 steps. walks < 2 steps may be something else 
    
    input: walklog - dict of walks that have start& end  time (seconds) and
    number of steps per walk 
        epoch : list of datetime object for the start and end of the epoch under examination
        
    #TODO may be better to have these indexed by time rather than sample numbers 
    which probably means changing the walklog to have start time, endtime rather 
    than seconds since the beginning 
    '''
   # walklog=walks.copy()
    # get min max for walk speeds to get rgb limits 
    ll = [(v[1]-v[0])/v[2]  for k,v in walklog.items() if v[2] > 2]
    # scale min-max for g=rgb 0.25 - 0.75
    vrange = max(ll) - min(ll)
    vmin = min(ll)
    ll2 = ((ll-vmin)/vrange)*0.5+0.25
    
   
    stime = pd.to_datetime(epoch[0])
    
    for walk,detail in walklog.items():
        if detail[2]> 2:  # walklength(no. steps)  > 2 
            print('walklog ',walk)
            stt = stime + timedelta(seconds=detail[0])
            ent = stime + timedelta(seconds=detail[1])
            x = mdates.date2num(stt)
            x2 = mdates.date2num(ent)
            y= -10
            w= (detail[1]-detail[0]) # length walk in sec
            h=20
            #x = 20000
            y = -10
           # w=20000
            steptime= w/(detail[2]) # in seconds
            #print(steptime,type(steptime))
            wt = x2 - x # width in matplotlib time format  
            rgbrval = ((steptime-vmin)/vrange)*0.8+0.2
          #  print(rgbrval,steptime,type(steptime))

            ax.add_patch(patches.Rectangle((x,y),wt,h
                                           ,fc=(rgbrval,0,0)
                                           ))
            
#            ax.text(x, y, 'center top',
#                    horizontalalignment='center',
#                    verticalalignment='top',
#                    transform=ax.transAxes)
            # print(x,w)
            print('hi')
            ax.text(x,-6.6,str(detail[2])+'('+str(walk)+')',color='lightgreen')
            ax.text(x,-7.5,'{0:1.2f}'.format(steptime),color='lightblue',
                    rotation=90)        
    out = ax.plot()
    return out

def plotPTTimes(ax,PTlog,epoch,rate=50,offset=0,param_dict={}):
    '''  plot pt times on axes 
    note that the x axis passed in has tod. the PTlog is in samples, so need to convert samples to timedelta 
    in seconds and add to epoch start '''
   
    h=20
    y=-10

    stime = pd.to_datetime(epoch[0])
    # PTs last between 0.5 and 4 seconds .. so throw away anything 
    # 1/2 the rate and 4 * rate- although are PTs contigous ? they are in FTSTS 
    for PT in (y for y in PTlog.values() if (y[2] >= 20)):
       # PT = PTlog[272] # 272, 341 
        x,x2 = PT[0]/50,PT[1]/50
        stt = stime + timedelta(seconds=x)
        ent = stime + timedelta(seconds=x2)
        x = mdates.date2num(stt)
        x2 = mdates.date2num(ent)
    
        # length walk in sec
        w= x2-x
        ax.add_patch(patches.Rectangle((x,y),w,h,
                                           fc='b',alpha=0.2))
        
    out = ax.plot
    return out
#plt.show()

def plotPTdetail(ax,PTdetail,rate=50,param_dict={}):
    ''' plot PT details on axes - intended to overlay the Vacc and sintheta 
    plots 
    
    PT detail consists of the classification of the PT, 3 x,y 
    co-ordinates of the sintheta peak and the vert velocity trough and peak 
    and the ratio of peak to trough. The ratio is used to classify the 
    transition '''
    # get first set of points
    PTclass = [x[0] for x in PTdetail.values()]
    sinth = [x[1] for x in PTdetail.values()]
    PTratio = [x[4] for x in PTdetail.values()]

    vvpeaks = [x[2] for x in PTdetail.values()]
    vvvalleys = [x[3] for x in PTdetail.values()]
    sinthpts = [[x[0] for x in sinth],[x[1] for x in sinth]]
    vvpeakspts = [[x[0] for x in vvpeaks],[x[1] for x in vvpeaks]]
    vvvalleyspts = [[x[0] for x in vvvalleys],[x[1] for x in vvvalleys]]
    ax.plot(sinthpts[0],sinthpts[1],'o', mfc=None, mew=2, ms=8,
             label='Sintheta peaks')
    ax.plot(vvpeakspts[0],vvpeakspts[1],'+', mfc=None, mew=2, ms=8,
             label='Vert Velocity peaks')
    ax.plot(vvvalleyspts[0],vvvalleyspts[1],'+', mfc=None, mew=2, ms=8,
             label='Vert Velocity valleys')

    
    for i,p in enumerate(PTclass):
        tb = ax.text(sinth[i][0],sinth[i][1]*1.1,str(p)+
                     ' '+'{0:1.2f}'.format(PTratio[i]))
        tb.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))
                

    
    out = ax.plot

    return out

def plotInact(ax, iasegs,rate=50,param_dict={}):
    '''  plot pt times on axes  '''
    h=4
    
    for ia in iasegs:
        x,x2 = (ia[0]-1)*rate,(ia[1])*rate
            
        w= x2-x + 1
        print(x,w,x)
        y=-2
        ax.add_patch(patches.Rectangle((x,y),w,h,
                                           fc='y',alpha=0.3))
    out = ax.plot
    return out

def plotPressure(ax, pressure,rate=25,param_dict={}):
    plt.plot(periodpressure.values,label='pressure')



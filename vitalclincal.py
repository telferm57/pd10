# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 05:14:55 2017

@author: telferm

analyse clinical test output 

given a participant id : 
    
get the file 

get on body segments - should only be one 

identify start ?

set start to  zero ? (or set annotation start to tod)

activity

try overlay 

walks

pts 

"""
import filehandler as fh
import imp
import numpy as np
import vitalsensor as vs 
import vitalplot1 as vp1
import matplotlib.pyplot as plt
from matplotlib import dates as mdates


#imp.reload(vs)
#imp.reload(vp1)
basedir = 'C:/Users/telferm/projects/vital/Batch1/'
def getMonitorData(subject,rdf,basedir,adl=0):
    
    fn = rdf[(rdf.subject==subject) & (rdf.adl == adl)].fn.values
    #should only be one 
    fn = fn[0]
    subdir = 'Batch1/'
    path = basedir + subdir
    
    #get matlabdata 
    dfsensor = fh.getMatlabfile(fn,path,subject)
    
    return dfsensor

   
#check sample rate 


def magnitude(ar): # assuming x, y, z array 
    ''' given an array with 3 columns or rows, return 1d array of magnitude 
    along the longest axis '''
    arsh = ar.shape  
    if 3 not in arsh:print('I only deal with 3d vectors - or do I ? ')
    if arsh[0]==3: long = 0 
    else: long = 1 
    if long: ar = ar.T
    ar2 = np.square(ar)
    m2 = ar2[0] + ar2[1] + ar2[2]
    m = np.sqrt(m2)
    return m




def countOneGroups(ar):
    ''' will count groupsof more than 1 1 in array apassed in 
    passes back number of groups 
    test : 
    ar =np.array([1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1])
    '''
    import re
    ar = 1*ar # convert to int if it already i'n't
    np.sum(ar)
    st = ''.join([str(e) for e in list(ar)])
    onegroups = re.findall('1+',st)
    return len(onegroups)


    
    # regex replace 010 with 0 
    
    #regex replace 0* with 0 
    # regex replace 1* with 1 
def getSpikes(mag,thresh=50):
    ''' return start and end of 5 - 10 spikes > mag thresh within 3 seconds
    1. get periods of > 50 
    2. find blocks of 3 secs with > 5 of them 
    3. figure it out '''
    spikes = np.greater(mag,thresh)
    spikel = []
    winsz = 200
    overlap = 150
    for  (start, end) in vs.windows(spikes,winsz,overlap):# window size of 5 secs, moving 50 samples 
         window = spikes[start:end]
         if (countOneGroups(window) > 4):
             if  bool(spikel):
                 print('found at ',start)# shrinks all 00 and 010 to 0 
                 if spikel[-1][1] > start:
                     #this is a continuation of previous interval 
                     spikel[-1][1] = start+winsz # update end of previous interval
                 else: # it's a new one
                     spikel.append([start,start+winsz])
             else: # first one 
                 spikel.append([start,start+winsz])
                 
    # TODO delineate end of window by looking at the last spike 
    # in the last window 
    spikel2=[]
    for aspike in spikel:
        # update start to first spike (1) after start 
        spikestart = aspike[0]
        newstart = spikestart + list(spikes[spikestart:]).index(True) # first 1 after spikestart 
        spikel2.append([newstart,aspike[1]])
        
    return spikel2



def fixupAnnot(annotations):
    annotdict = {'sync sensor':'sync',
                 'sync of sensors':'sync',
                 '6m walk normal speed':'6mw normal',
                 '6m walk fast speed':'6mw fast',
                 'stand on one leg right':'stand on R',
                 'stand on one leg left':'stand on L',
            }
   
    annotations['notes2'] = annotations['notes'].map(lambda x: annotdict.setdefault(x.lower(),x))
    
        
    return annotations

def setAnnotStartTime(data,annotations,spikel):
    ''' sets start time (tod)  on annotations matched to the start of the 
    synch point detected in the sensor data '''
    from datetime import timedelta, datetime
 
    # add time index for dataframe 
    annotations['stime']=annotations['start'].map(lambda x : datetime.strptime(x,'%H:%M:%S.%f'))
    # add a small time  to step count times so they do not overlap 
    delta1sec = timedelta(seconds=0.5)
    annotations['stime'] = annotations.apply(lambda x: x['stime']+ delta1sec if  x['section'] == 'no. steps' else x.stime,
               axis=1)
    
    annotations['delta'] = annotations['stime'].map(lambda x: timedelta(hours=x.hour,
               minutes=x.minute, seconds=x.second, microseconds=x.microsecond))
    # get start time of first synch from signal data 
    startsynch = data.datetime.iloc[spikel[0][0]]
    # get annotations start of synch - there are usually 2 synchs , we want the earliest
    annotsynch = annotations[annotations.notes2=='sync'].stime.min()
    #change it to a delta 
    synchdelta = timedelta(hours=annotsynch.hour,
               minutes=annotsynch.minute, seconds=annotsynch.second,
               microseconds=annotsynch.microsecond)
    # tod = tod from signal + delta from annotations - delta from synch in annotations
    annotations['synched'] = annotations['delta'].map(lambda x: startsynch+x-synchdelta)
    
    #sort annotations 
    return annotations
#%%
 
subject = 'vah010'
subject = 'vah001'
subject = 'vah006'
basedir = 'C:/Users/telferm/projects/vital/data/'
datasubir = 'Batch1/'
#TODO rdf is created by create file report in filehandler ... but it is an internal
# variable for that function
#       imp.reload(vs) 
report, rdf = fh.createFilereport(basedir,datasubir)
dfsensor = getMonitorData(subject,rdf,basedir)
samplerate  =len(dfsensor)/(dfsensor.timestamp.max()-dfsensor.timestamp.min())
#samplerate = np.around(samplerate,decimals=1)
onbodywindow = 2 # minutes window size 
       #TODO standardise all times in seconds 
onbodyperiods, offBodyPeriods = vs.dissectData(dfsensor,onbodywindow)

for period in onbodyperiods.values():
    #TODO modify for multiple periods 
    startperiod = int(period[0]*60*onbodywindow*samplerate)
    endperiod = int((period[1]+1)*60*onbodywindow*samplerate)

ondata = dfsensor.iloc[startperiod:endperiod,:]
gravlist = vs.getGravity(dfsensor,offBodyPeriods,onbodywindow,samplerate)
gmean = np.nanmean(gravlist)
gravity = 9.81 #TODO use derived gravity above .. need to assess the effect 
# note - following vectors are length 7 less than input due to 
# adjustment for lag after filtering 

ar, arb, argr, ars, vacc, Vhacc, newdf = vs.getVectors(ondata,startperiod,endperiod,gvalue=gravity)

epoch = [newdf.datetime.values[0],newdf.datetime.values[-1]] #TODO these are np format - y?
mag = vs.magnitude(ar)
#plt.plot(ar)
spikel = getSpikes(mag,thresh=25) # match with sync from annotations
annotations = fh.getAnnotations(subject,basedir) #
annotations = fixupAnnot(annotations)
annotations = setAnnotStartTime(newdf,annotations,spikel)
# we now have annoations with timings aligned with the datetime of the sensor data 
# get walks - first we need the sma values to then  
smawindow = 50 # first we need the sma values to then...  
sma = vs.getSMA(arb, winsize=smawindow)
#TODO check reasonableness for SMA 

actclasses=vs.getActivityClasses(sma,g=gravity) #... compute the activity classes 
# sma has one value per window of 1 second. so  timestamps are pos * 50
smat =  np.arange(0,len(sma),1)*50
imp.reload(vs)
sinfilt = vs.movingAv(sintheta,12)
#get inactive segments - this is also called by sintheta to reset vertical as 
# frequently as possible  .  
plt.plot(sinfilt)
plt.plot(Vv)
iasegs = vs.getInactiveSegments(sma,5,1.5)
#sinfilt[51380:51390]
plt.plot(smat,sma)
# get the walks 
walks = vs.getWalks(vacc,actclasses,smawindow,epoch)  
# get postural transitions 
sintheta = vs.getSintheta(ar,sma) # angle of device with vertical 

PTlog = vs.getPT(walks,sintheta,samplerate,angleth=0.40)

PTdetail = vs.getPTdetail(vacc,sintheta,PTlog,smph=0.40)
# pressure - create normalised 

def adjustPressure(newdf):
    ndf = newdf.copy()
    pressfilt = medfilt(ndf.press.values,25)
    pressmin = ndf.press.min()
    pressmax = ndf.press.max()
    pressmean = ndf.press.mean()
    pressrange=pressmax-pressmin
    ndf['presstd'] = ndf['press'].map(lambda x: ((x-pressmean)/(pressrange/2) ))
    
    return ndf



#%% Gait Analysis for each segment in minibest 
def getNearestWalk(stime,walks):
   # stime = annwalks.iloc[0].synched
    # get vector of starttimes
    stimes  =  [x[4] for x in walks.values()]
    # adjust for time differences betwen annotation and walks# stime = annwalk
    stimesdiff = [abs((date -stime).total_seconds()) for date in stimes]
    
    return stimesdiff.index(min(stimesdiff)) + 1 # walks index starts at 1 
   
imp.reload(vs)
    #subtract this one 
    # get smallest value 
    
dfgait = vs.getGaitfeatures(walks,arb,samplerate=samplerate)   
gaitcolumns=['subject','test','start','end',
             'steps1','avg_step_duration','cadence',
             'steps2','sd_step','sd_stride',
             'step_regularity', 'stride_regularity','symmetry'] 
clinical_visit_gait = pd.DataFrame(columns=gaitcolumns)
    
minibest = ['6mw normal','6mw fast','gait speed normal',
            'walk with head turns','walk with pivot turns (walk)']

for test in minibest:
    #get start and end
        #get times from annot
       # test = minibest[0]
        annwalks = annotations[annotations.notes2==test]
#        for each one (can be multiple):
        for annwalk in annwalks.synched:
            sensorwalk = getNearestWalk(annwalk,walks)
            print('nearest walk: ',test,annwalk,sensorwalk)
            print('gait report:', dfgait[dfgait.walkid==sensorwalk].number_of_steps)
            
            dfgaitsr = dfgait[dfgait.walkid==sensorwalk] 
            start_time = walks[sensorwalk][4]
            
            if len(dfgaitsr) == 0: 
                gait_row = [subject,test,annwalk,start_time,0,
                        0,0,0,0,0,0,0,0]
            else:
                gait_row = [subject,test,annwalk,start_time,
                            dfgaitsr.Method1_stepcount[0],
                            dfgaitsr.avg_step_duration[0],dfgaitsr.cadence[0],
                 dfgaitsr.number_of_steps[0],dfgaitsr.sd_step_durations[0],
                 dfgaitsr.sd_stride_durations[0],
                 dfgaitsr.step_regularity[0],dfgaitsr.stride_regularity[0],
                 dfgaitsr.symmetry[0]]
                
            gait_dict = {x:y for (x,y) in zip(gaitcolumns,gait_row)}
            clinical_visit_gait = clinical_visit_gait.append(gait_dict,ignore_index=True)
            
            #locate nearest walk in walks 
            #get gait features 
    
    produce report

#%% plot PTs alone 
fig1 = plt.figure()
ax = fig1.add_subplot(111)
#plt.plot(Vvert,label='Vert Vel')
#create x from datetime column of newdf created manually at the moment above
x = pd.to_datetime(newdf.datetime.values)

x2 = np.array(list(map(mdates.date2num,x)))
Vv = vs.getVertVelocity(vacc)
plt.plot(x2,Vv,label='Vert Vel')
vp1.plotPTTimes(ax,PTlog,epoch) # need to align with annotations, hence x2 
sinfilt = vs.movingAv(sintheta,20)
#plt.plot(x2,sintheta, label = 'sintheta filtered 11')
plt.plot(x2,sinfilt, label = 'sintheta filtered moving average 20',color='g')
#vp1.plotPTdetail(ax,PTdetail,epoch,50,{})
annotax,axx  = vp1.pltAnnotations(ax,annotations)
newdf = adjustPressure(newdf)
plt.plot(x2,newdf.presstd)
plt.show()
plt.legend()

#%% plot walks alone
fig1 = plt.figure()
ax = fig1.add_subplot(111)
#plt.plot(Vvert,label='Vert Vel')
#create x from datetime column of newdf created manually at the moment above
x = pd.to_datetime(newdf.datetime.values)

x2 = np.array(list(map(mdates.date2num,x)))
Vv = vs.getVertVelocity(vacc)
ax.plot(x2,Vv,label='Vert Vel')

ax.plot(x2,vacc,label='Vert Acc')
vp1.plotWalkTimes(ax,walks,epoch,samplerate,{'label':'walks'})
timeFmt = mdates.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_formatter(timeFmt)
fig.autofmt_xdate()
annotax,axx  = pltAnnotations(ax,annotations)
#newdf = adjustPressure(newdf)
#plt.plot(x2,newdf.presstd)
plt.show()
ax.legend()
#%% plotting them  
import vitalplot1 as vp1
imp.reload(vp1)
fig1 = plt.figure()
ax = fig1.add_subplot(111)
#plt.plot(Vvert,label='Vert Vel')
#create x from datetime column of newdf created manually at the moment above
x = pd.to_datetime(newdf.datetime.values)

x2 = np.array(list(map(mdates.date2num,x)))

plt.plot(x2,vacc,label='Vert Acc')
Vv = vs.getVertVelocity(vacc)
plt.plot(x2,Vv,label='Vert Vel')
vp1.addGrid()
vp1.addGrid()
plotWalkTimes(ax,walks,epoch,samplerate,{'label':'walks'})
timeFmt = mdates.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_formatter(timeFmt)
fig.autofmt_xdate()
vp1.plotPTTimes(ax,PTlog,epoch) # need to align with annotations, hence x2 
vp1.plotInact(ax,iasegs)
#sinfilt = medfilt(sintheta,11)


sinfilt = vs.movingAv(sintheta,20)
#plt.plot(x2,sintheta, label = 'sintheta filtered 11')
plt.plot(x2,sinfilt, label = 'sintheta filtered moving average 20',color='g')
vp1.plotPTdetail(ax,PTdetail,epoch,50,{})
annotax,axx  = pltAnnotations(ax,annotations)
ax.plot()
plt.plot(x2,newdf.presstd.values,label='pressure') 
plt.show()
plt.legend()

#%% read and plot annotation data 
#TODO put timestamps on these                
smafiles,vectorfiles = multifileSMA(reg,basedir,subjects=['vah002'])

activityLevels = multifileActclasses(smafiles,gravity=9.81)

#times = mdates.date2num(tst.values)
#convert synched  annot times to matplotlib 

fig, ax = plt.subplots()



# plot vacc for trial  
plt.plot(newdf['datetime'].values,vacc)
plt.axhline(y=0)
plt.axhline(y=1.4,c='r')
plt.plot([],[])
#plt.scatter(tst.values,np.random.random(69))

#fig.autofmt_xdate() too complex
ax.minorticks_on()
# dont use grid use vertical lines 
ax.grid(which='both')
#plt.gcf().set_size_inches(32,24)

fig,ax = plt.plot()

        
#%% plot 'monitor active' times       

from datetime import time,datetime, date

x = rdf.date.apply(lambda x: datetime.strptime(x,'%Y-%m-%d').date())
y = rdf.start.apply(lambda x: datetime.strptime(x,'%H:%M:%S').time())
end = rdf.end.apply(lambda x: datetime.strptime(x,'%H:%M:%S').time())

#plt.plot_date(x,y,ydate=True)
#plt.plot_date(x,end,ydate=True,c='r')
for i in range(len(x)):
    plt.plot_date([x[i],x[i]],[y[i],end[i]],'ro-')

plt.axvspan(x[3],x[4],fc='b')
#plot acivity levels 

data = np.arange(100, 0, -1).reshape(10, 10)

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])

im = ax.imshow(data, cmap='gist_earth')
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.show()
a1 = activityLevels['vah002']['vah002h2800A096236B0F.mat']

x = np.arange(6)
y = np.arange(5)
z = x * y[:, np.newaxis]

#%% sample heatmap based on date/time
# generate a range of times over 2 dates 
tt = []
for h in range(8):
    #print(type(tt))
    for m in np.arange(0,46,15):
        tt.append(datetime(2017,1,18,h,m,0))

ff = np.random.random(64) # generate  random values  
dftt = pd.DataFrame(data=ff,index=tt)   
 
# get unique dates (i.e. date/month/year) from data 
dddates = np.array([date.isoformat(x) for x in dftt.index.date])
dddates = list(set(dddates))
dddates.sort(reverse=True)

# put different dates data into columns 
datear={}   
for dtsa in dddates:
   s1 = dftt[dtsa][0] # get series 
   s1.index = s1.index.time
   datear[dtsa] = s1
   
s1 = dftt[]   
dtff2 = pd.DataFrame.from_dict(datear)
# plot it 
# https://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
fig, ax = plt.subplots()
cmap = plt.get_cmap('Blues')
# see https://stackoverflow.com/questions/9214971/cmap-set-bad-not-showing-any-effect-with-pcolor
# for handling nans in the data 
cmap.set_bad(color = 'k', alpha = 1.)
heatmap = ax.pcolormesh(dtff2, cmap=cmap, alpha=0.8)
# make some cells nan (where there is no data )
dtff2.loc[(dtff2.index < time(5,30,0)) & (dtff2.index > time(2,30,0) ) ]

# put the major ticks at the middle of each cell

ax.set_yticks(np.arange(dtff2.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(dtff2.shape[1]) + 0.5, minor=False)
ax.set_xticklabels(dtff2.columns, minor=False)
ax.set_yticklabels(dtff2.index, minor=False)



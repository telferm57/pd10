# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 12:21:27 2017

@author: Malcolm Telfer

adapted from: 
    http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass

process standing signals 
"""

from scipy.signal import butter, lfilter, welch, periodogram, find_peaks_cwt
from numpy.fft import rfft,irfft, fftshift, ifftshift
from mhealthx.signals import compute_sample_rate
from mhealthx.xio import read_accel_json
import pandas as pd
import numpy as np
from BMC.detect_peaks import detect_peaks


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band',analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    cutf = cutoff / nyq
    b, a = butter(order, cutf, btype='low',analog=False)
    return b, a

def lpFilter(x,y,z,cutoff,samplerate):
    xfilt = butter_lowpass_filter(x,cutoff,samplerate,4)
    yfilt = butter_lowpass_filter(y,cutoff,samplerate,4)
    zfilt = butter_lowpass_filter(z,cutoff,samplerate,4)
    return xfilt,yfilt,zfilt

def magnitude(ar): # assuming x, y, z array 
  
    ar2 = np.square(ar)
    m2 = ar2[0] + ar2[1]+ar2[2]
    m = np.sqrt(m2)
    return m

def calcPower(data,t):
    ''' data is array of x, y, z values '''
 
    power = (np.sum(data**2,axis=1))/(2*len(t))
    return power

def calcEntropy(data,samplerate,welchMethod=False):
   
    from scipy.stats import entropy
    if welchMethod:
        f, Pxx_den = welch(data, samplerate,nperseg=400) # approx 2 sec 
    else:
        f, Pxx_den = periodogram(data, samplerate)
    
    pdf = Pxx_den/Pxx_den.sum()    
   
    return entropy(pdf)

def getTestData(fn,deviceMotion = False):
    path='F:\\DS\\diss\\data\\testfiles\\' + fn
    if deviceMotion:
        t, axyz, gxyz, wxyz, rxyz,  \
        sample_rate, duration = read_accel_json(path, 0, True)
        #axyz is list of lists of user (i.e. no gravity) acceleration 
        npa = np.array(axyz)
        npa = npa.transpose()
        restDf  = pd.DataFrame(data=npa,columns=['x','y','z'])
        restDf['timestamp']= t
        return restDf 
    else:
        restDf = pd.read_json(path)
        return restDf

def calcJerkSignal(data,samplerate):
    '''calculates the jerk signal as three vectors, returns those + magnitude '''
    bpv = butter_lowpass_filter(data,3.5,samplerate,4)
    jerksig = np.diff(bpv)# len 1 less that input 
    m = magnitude(jerksig)
    window = np.ones(100)/100
    msmooth = np.convolve(m,window,mode='same') # smooth signal 
    return jerksig, msmooth

def getPeaks(data,samplerate):
    ''' get peak info for signal:
        mean and sd of peak amplitude
        peak frequency
        peak height to width '''
    #peakind = find_peaks_cwt(data, np.arange(1,10))
    bmcpeakind = detect_peaks(data,mpd=50,show=False)
    jpeaks = len(bmcpeakind)
    jfreq = jpeaks/(len(data)/samplerate)
    amps = data[bmcpeakind]
    jmean = amps.mean()
    jstd = amps.std()
    return jfreq,jmean,jstd
#%%   
def getRestFeatures(restDf,healthCode,includeAudio,deviceFile):


    print('getRestFeatures hc: ',healthCode)
   # fn = 'accel_walking_rest.json.items-updrs2_10_3.json'
   # path='F:\\DS\\diss\\data\\testfiles\\' + fn
   # restDf = pd.read_json(path)
    
    #import pdb
   # pdb.set_trace()
    ts = restDf.timestamp
    ts -= np.min(ts)
    # for device files, user acceleration is in cols userAcceleration_x, y, z
    if deviceFile:
        restDf.rename(columns={'userAcceleration_x':'x',
                               'userAcceleration_y':'y',
                               'userAcceleration_z':'z'},inplace=True)
    
    samplerate, duration = compute_sample_rate(ts.values)
    x = restDf.x.values[300:-300] # to remove settling values 
    y = restDf.y.values[300:-300]
    z = restDf.z.values[300:-300]
    ts = ts[300:-300]
    if len(ts) < 500: return ['tooshort'],len(ts)

    filteredx = butter_bandpass_filter(x,4,7,samplerate,4)
    filteredy = butter_bandpass_filter(y,4,7,samplerate,4)
    filteredz = butter_bandpass_filter(z,4,7,samplerate,4)
    
   
    dataPreFiltered = np.array([x,y,z])
    dataFiltered = np.array([filteredx,filteredy,filteredz])
    powerPre = calcPower(dataPreFiltered,ts)
    powerFiltered = calcPower(dataFiltered,ts)
    HR1 = powerFiltered/powerPre # ratios along each axis 
    HR1sum = HR1.sum()
   # filter data < 3.5 to remove tremor (see Palmerini ) 
    
    lpfilteredx = butter_lowpass_filter(x,3.5,samplerate,4)
    lpfilteredy = butter_lowpass_filter(y,3.5,samplerate,4)
    lpfilteredz = butter_lowpass_filter(z,3.5,samplerate,4)
                                        
    lpfilteredx,lpfilteredy,lpfilteredz = lpFilter(x,y,z,3.5,samplerate)
    # aproximate gravity signal 
    gfilteredx,gfilteredy,gfilteredz = lpFilter(x,y,z,0.3,samplerate)
   # remove gravity 
    xNog = x-gfilteredx
    yNog = y-gfilteredy  
    zNog = z-gfilteredy

    xNogFiltered,yNogFiltered,zNogFiltered =  lpFilter(xNog,yNog,zNog,3.5,samplerate)
    # with gravity : 
    entropyx = calcEntropy(lpfilteredx,samplerate,True)
    entropyy = calcEntropy(lpfilteredy,samplerate,True)
    entropyz = calcEntropy(lpfilteredz,samplerate,True)
    # without 'gravity' : 
    entropyxNog = calcEntropy(xNog,samplerate,True)
    entropyyNog = calcEntropy(yNog,samplerate,True)
    entropyzNog = calcEntropy(zNog,samplerate,True)
    
    # take max entropy - axes are not aligned with anatomical axes 
    maxEntropy = max([entropyxNog,entropyyNog,entropyzNog])
    rmsEntropy = (np.array([entropyxNog,entropyyNog,entropyzNog])**2).sum()**0.5
    
    s = dataPreFiltered # allow jerk to do its own filtering 
   
    # get rid of tremor with low pass filter
    jxyz, ms = calcJerkSignal(s,samplerate) 
    
    jfreq,jmean,jstd = getPeaks(ms[1000:],samplerate)




    # print('age:',restDf.age[1],'diagnosis:',restDf.professionalDiagnosis[1],
     #     'updrs2_10:',restDf.updrs2_10[1],'HR:',np.sum(powerFiltered/powerPre),
      #    powerFiltered/powerPre,'ent x,y,z',maxEntropy)
    
    return {'HR1':[HR1sum],'maxEntropy':[maxEntropy],'rmsEntropy':rmsEntropy,
            'jfreq':jfreq,'jmean':jmean,'jstd':jstd}
#%%
# https://dsp.stackexchange.com/questions/34458/discussion-integrating-accelerometer-data-to-position-data-from-frequency-doma
 # l = length(a); % assume l even 
 #w = 2*pi*fs/l*(-l/2:l/2-1);
 #A = fftshift(fft(a));
 #D = A./-w.^2;
 #d = ifft(ifftshift(D),'symmetric');
 
def calcDisplacement(data,fs):
    
    ''' funtion not return expected results - abandoning for now '''  
    data =y.copy()
    fs = samplerate.copy()
    l = len(data)
    if l//2 !=0:
        data = data[:-1]
    l -= 1
       
    w = (2*np.pi*fs/l)*np.arange(1,(l/2+2))
    len(w)
    A = fftshift(rfft(data))
    len(A)
    D = A/-(w**2)
    d = irfft(ifftshift(D))
    return d
 
 # plot d 
    limits = 0 
    title = 'wed'
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(ts, z)
    plt.subplot(2, 1, 2)
    plt.plot(ts, lpfilteredz)
   #
   


#Duarte, M. (2015) Notes on Scientific Computing for Biomechanics and Motor Control. 
#GitHub repository, https://github.com/demotu/BMC.

fn = 'accel_walking_rest.json.items-updrs2_10_3.json'
fn2 = 'accel_walking_rest.json.items-updrs2_10_2_1.json'
fn3 = 'accel_walking_rest.json.items-updrs2_10_2_2.json'
fn4 = 'deviceMotion_walking_rest.json.items-updrs2_10_2.json'

data = getTestData(fn3,False)





        
def restPlotit(ts,x,y,z,filteredx,filteredy,filteredz,title):
    
    limits = 0 
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(ts, x)
    if limits:
        plt.ylim((limits[0], limits[1]))
    plt.title('x-axis unfiltered ' + title)
    plt.ylabel(title)
    
    plt.subplot(3, 2, 2)
    plt.plot(ts, filteredx)
    if limits:
        plt.ylim((limits[0], limits[1]))
    plt.title('x-axis filtered ' + title)
    plt.ylabel(title)
    
    plt.subplot(3, 2, 3)
    plt.plot(ts, y)
    if limits:
        plt.ylim((limits[0], limits[1]))
    plt.title('y-axis ' + title)
    plt.ylabel(title)
    
    plt.subplot(3, 2, 4)
    plt.plot(ts, filteredy)
    if limits:
        plt.ylim((limits[0], limits[1]))
    plt.title('y-axis filtered  ' + title)
    plt.ylabel(title)
    
    plt.subplot(3, 2, 5)
    plt.plot(ts, z)
    if limits:
        plt.ylim((limits[0], limits[1]))
    plt.title('z-axis ' + title)
    plt.xlabel('Time (s)')
    plt.ylabel(title)
    
    plt.subplot(3, 2, 6)
    plt.plot(ts, filteredz)
    if limits:
        plt.ylim((limits[0], limits[1]))
    plt.title('z-axis filtered ' + title)
    plt.xlabel('Time (s)')
    plt.ylabel(title)
    plt.show()
        
    
    
    plt.plot(ts, filtered)


    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.show()
   
   


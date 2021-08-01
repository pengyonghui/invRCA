# -*- coding: utf-8 -*-

"""
Created on Sun Sep 13 13:54:59 2019

@author: yonghui

non-negative least squares(NNLS) using tikhonov regularization for nmr data analysis
- first written by Elliot Grunewald in Sep. 2006 
- Keating and Falzone revised on 08/31/2012

"""


import numpy as np
import math 
from scipy.optimize import nnls

def T2NNLS(data,time,noise,T2,eps, winsize):
    """   
    Input:
    data: data vector corresponding to 'time', NO time ZERO allowed!!
    time: time vector corresponding to 'data'
    noise: noise vector corresponding to 'time' and 'data'
    T2: vector of available T2 values to which the distribution is fit
    eps: regularization factor
    winsize: the size to clip the data
    
    Output:
    m: inverted model 1, length(T2)+1 with m(length(T2)+1) = baseline offset 
    dsyn:  dysn(:,0) resampled time , dysn(:,1) modeled data M(time), and dysn(:,2) resampled data
    r: reidual norm and data tol r(0) = residual norm; r(1) = model norm

    """

    stdev = np.std(noise)
    n = 20
    
    ## clip decayed data by finding when moving average reaches some fraction of initial signal 
    # winsize = 100 
    filt = 1/winsize*np.ones(winsize)
    filt_d_conv = np.where(np.convolve(filt,data) < data[0]/1e5)[0]
    if filt_d_conv.size > 0:
        end_decay = min(filt_d_conv) 
    else:
        end_decay = 0
    if end_decay < 1 or end_decay > len(data): end_decay = len(data)
    if end_decay <= winsize:  end_decay = len(data)
    
    data = data[0:end_decay]
    time = time[0:end_decay]
    noise = noise[0:end_decay]
    
    if end_decay >= 1000:
        #the ideal log spacing of data
        time_log_ideal = np.logspace(math.log10(time[0]),math.log10(time[-1]), 
                                num = math.floor(len(time)/n), endpoint = True,dtype = float) 

        # time_log_ideal = time_log_ideal[::-1] 
        temp_time = np.zeros(len(time_log_ideal))
        temp_data = np.zeros(len(time_log_ideal))
        temp_stdev = np.zeros(len(time_log_ideal))
#        temp_data = data[0]
#        temp_stdev = stdev
        temp_time[0] = time[0]
        temp_data[0] = data[0]
        temp_stdev[0] = stdev
        
        lastindex = 0
        for i in range(1,len(time_log_ideal)):
            cindex = np.argmin(np.abs(time - time_log_ideal[i]))
            if cindex == lastindex:
                temp_time[i] = np.nan
                temp_data[i] = np.nan
                temp_stdev[i] = np.nan
            else:
                temp_time[i] = np.mean(time[lastindex + 1:cindex + 1])
                temp_data[i] = np.mean(data[lastindex + 1:cindex + 1])
                temp_stdev[i] = stdev/np.sqrt(cindex - lastindex)
            lastindex = cindex 
            
        nonnan = np.where(~np.isnan(temp_time))[0]
        resampled_time = temp_time[nonnan]
        reampled_data_nonweighted = temp_data[nonnan]
        rstdev = temp_stdev[nonnan]
        resampled_data_weighted = reampled_data_nonweighted/rstdev
        
    else:
        resampled_time = time
        rstdev = stdev*np.ones(len(data))
        reampled_data_nonweighted = data
        resampled_data_weighted = data/rstdev
        
    ##-------------------------- setup kernel matrix (Y)
    G = np.zeros((len(resampled_time),len(T2) + 1))
    Lwoweightreg = np.zeros((len(resampled_time), len(T2) + 1))  

    for i_Lindex in range(len(resampled_time)):
        for j_Lindex in range(len(T2)):
            Lwoweightreg[i_Lindex,j_Lindex] = np.exp(-resampled_time[i_Lindex]/T2[j_Lindex])
            G[i_Lindex,j_Lindex] = np.exp(-resampled_time[i_Lindex]/T2[j_Lindex])/rstdev[i_Lindex]
            
        Lwoweightreg[i_Lindex, len(T2)] = 1
        G[i_Lindex, len(T2)] = 1/rstdev[i_Lindex]
                    
    ##-------------------------- creat regularization Matrix (Y)
    k = len(T2) + 1 # number of columns in G
    reg = np.diag(-2*np.ones(k), 0) + np.diag(np.ones(k-1), 1) + np.diag(np.ones(k-1), -1)
    reg = np.concatenate([reg[1:k-2, 0:k-1],np.zeros((k-3,1))], axis=1)
    reg = np.concatenate(([np.append([1],np.zeros(k-1))],[np.append([-2,1],np.zeros(k-2))], reg,
                         [np.append(np.zeros(k-3),[1,-2,1])]), axis=0) 

    Lreg = np.vstack((G, eps*reg))                           
    ## non-negative lsq inversion
    lsq_rd = np.append(resampled_data_weighted, np.zeros(k)) 
   
    m, rnorm = nnls(Lreg, lsq_rd)
    
    dsyn = np.zeros((len(resampled_time),3))
    r =np.zeros(2)
#    
    dsyn[:,0] = resampled_time    # resampled time
    dsyn[:,1] = np.dot(Lwoweightreg, m) # fitted data
    dsyn[:,2] = reampled_data_nonweighted# resampled data
   # dsyn[:,2] = np.subtract(reampled_data_nonweighted, dsyn[:,1])# residuals 
    # residual_norm = np.dot(G, m)
    r[0] = np.linalg.norm(np.subtract(dsyn[:,2], dsyn[:,1])) # residual norm
   # r[1] = np.linalg.norm(rstdev) #data tol
    #r[0] = rnorm
    r[1] = np.linalg.norm(m)# model norm
    return (m, r, dsyn, rnorm)
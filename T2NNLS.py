# -*- coding: utf-8 -*-

"""
Created on Sun Sep 13 13:54:59 2019

@author: yonghui

non-negative least squares(NNLS) using tikhonov regularization for nmr data analysis
- first written by Elliot Grunewald in Sep. 2006 
- Keating and Falzone revised on 08/31/2012

"""


import numpy as np
import cmath 
from scipy.optimize import nnls

def T2NNLS(d,t,noise,T2,eps, winsize):
    """   
    Input:
    d: data vector corresponding to t, NO time ZERO allowed!!
    t: time vector corresponding to d
    noise: noise vector corresponding to t and d
    T2: vector of available T2 values to which the distribution is fit
    eps: regularization factor
    winsize: the size to clip the data
    
    Output:
    m: inverted model 1, length(T2)+1 with m(length(T2)+1) = baseline offset 
    r: reidual norm and data tol r(0) = residual norm; r(1) = model norm
    dsyn:  dysn(:,0) resampled t , dysn(:,1) modeled data M(t), and dysn(:,2) resampled data

    """

    stdev = np.std(noise)
    n = 20
    
    ## clip decayed data by finding when moving average reaches some fraction of initial signal 
    # winsize = 100 
    filt = 1/winsize*np.ones(winsize)
    filt_d_conv = np.where(np.convolve(filt,d) < d[0]/1e5)[0]
    if filt_d_conv.size > 0:
        enddecay = min(filt_d_conv) 
    else:
        enddecay = 0
    if enddecay < 1 or enddecay > len(d): enddecay = len(d)
    if enddecay <= winsize:  enddecay = len(d)
    
    d = d[0:enddecay]
    t = t[0:enddecay]
    noise = noise[0:enddecay]
    
    if enddecay >= 1000:
        tlog_ideal = np.logspace(cmath.log10(t[0]),cmath.log10(t[len(t) - 1]), len(t)/n, endpoint = True,dtype = float)
        # tlog_ideal = tlog_ideal[::-1] # reverse order so goes from small to large
        tempt = np.zeros(len(tlog_ideal))
        tempd = np.zeros(len(tlog_ideal))
        tempstdev = np.zeros(len(tlog_ideal))
#        tempd = d[0]
#        tempstdev = stdev
        tempt[0] = t[0]
        tempd[0] = d[0]
        tempstdev[0] = stdev
        
        lastindex = 0
        for i in range(1,len(tlog_ideal)):
            cindex = np.argmin(np.abs(t - tlog_ideal[i]))
            if cindex == lastindex:
                tempt[i] = np.nan
                tempd[i] = np.nan
                tempstdev[i] = np.nan
            else:
                tempt[i] = np.mean(t[lastindex + 1:cindex + 1])
                tempd[i] = np.mean(d[lastindex + 1:cindex + 1])
                tempstdev[i] = stdev/np.sqrt(cindex - lastindex)
            lastindex = cindex 
            
        nonnan = np.where(~np.isnan(tempt))[0]
        rt = tempt[nonnan]
        rd = tempd[nonnan]
        resampleddata = rd
        rstdev = tempstdev[nonnan]
        rd = rd/rstdev
        
    else:
        rt = t
        rstdev = stdev*np.ones(len(d))
        resampleddata = d
        rd = d/rstdev
        
    ##-------------------------- setup kernel matrix (Y)
    G = np.zeros((len(rt),len(T2) + 1))
    Lwoweightreg = np.zeros((len(rt), len(T2) + 1))  

    for i_Lindex in range(len(rt)):
        for j_Lindex in range(len(T2)):
            Lwoweightreg[i_Lindex,j_Lindex] = np.exp(-rt[i_Lindex]/T2[j_Lindex])
            G[i_Lindex,j_Lindex] = np.exp(-rt[i_Lindex]/T2[j_Lindex])/rstdev[i_Lindex]
            
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
    lsq_rd = np.append(rd, np.zeros(k)) 
   
    m, rnorm = nnls(Lreg, lsq_rd)
    
    dsyn = np.zeros((len(rt),3))
    r =np.zeros(2)
#    
    dsyn[:,0] = rt    # resampled time
    dsyn[:,1] = np.dot(Lwoweightreg, m) # modeled data
    dsyn[:,2] = resampleddata# 
   # dsyn[:,2] = np.subtract(resampleddata, dsyn[:,1])# residuals    
   # r[0] = np.linalg.norm(np.subtract(resampleddata, dsyn[:,1])) # residual norm
   # r[1] = np.linalg.norm(rstdev) #data tol
    r[0] = rnorm
    r[1] = np.linalg.norm(m)# model norm
    return (m,r,dsyn)
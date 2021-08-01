# -*- coding: utf-8 -*-

"""
Created on Sun Sep 13 13:54:59 2019

@author: yonghui

Control func that fits NMR data multiexponentially using non-negative least squares(NNLS)
with tikhonov regularization for nmr data analysis

- first written by Elliot Grunewald in Sep. 2006 
- Keating and Falzone revised on 08/31/2012

"""


import numpy as np
import math 
import numpy.matlib as mat
from scipy.optimize import nnls

def T2NNLS(data,time,noise,T2,eps, winsize=100):
    """
    [m, dsyn, r]  = T2NNLS(d,t,noise,T2,eps)
    
    Input:
    data: data vector corresponding to 'time', NO time ZERO allowed!!
    time: time vector corresponding to 'data'
    noise: noise vector corresponding to 'time' and 'data'
    T2: vector of available T2 values to which the distribution is fit
    eps: regularization factor
    winsize: the size to clip the data

    Output: 
    m: inverted model 1 (amplitude of each T2), length(T2)+1 with m(length(T2)+1) = baseline offset 
    dsyn:  dysn(:,0) resampled time , dysn(:,1) modeled data M(time), and dysn(:,2) resampled data
    r: reidual norm and data tol r(0) = residual norm; r(1) = model norm

    """

    stdev = np.std(noise)
    n = 20 # down sampling
    
    ## clip decayed data by finding when moving average reaches some fraction of initial signal 
    #winsize = 100 
    # create moving average filter
    filt = 1/winsize*np.ones(winsize)
    # find when move average < 1e-5 of the first 'data'
    filt_d_conv = np.where(np.convolve(filt,data) < data[0]/1e5)[0]

    if filt_d_conv.size > 0:
        end_decay = min(filt_d_conv) 
    else:
        end_decay = 0
    if end_decay < 1 or end_decay > len(data): end_decay = len(data)
    if end_decay <= winsize: end_decay = len(data)
    
    data = data[0:end_decay+1]
    time = time[0:end_decay+1]
    noise = noise[0:end_decay+1]
    
    # reduce sampling-log sample
    # resampling data
    if end_decay >= 1000:
        #the ideal log spacing of data
        #print(f"time[0] is {time[0]}, time[-1] is {time[-1]}")
        time_log_ideal = np.logspace(math.log10(time[0]), math.log10(time[-1]), num = len(time)//n, 
                                     endpoint = True, dtype = float) 

        #time_log_ideal = time_log_ideal[::-1] 
        temp_time = np.zeros(len(time_log_ideal))
        temp_data = np.zeros(len(time_log_ideal))
        temp_stdev = np.zeros(len(time_log_ideal))
        #temp_data = data[0]
        #temp_stdev = stdev
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
        
    ##-------------------------- setup kernel matrix 
    tempMatrix = createKernelMatrix(resampled_time, T2)
    G = np.ones((len(resampled_time),len(T2)+1))
    Lwoweightreg = np.ones((len(resampled_time),len(T2)+1))
    Lwoweightreg[:, :-1] = tempMatrix 
    G[:, :-1] = tempMatrix 
    G = G/rstdev[:, None] # or rstdev[:,np.newaxis] also works   
                       
    ##-------------------------- creat regularization Matrix 
    N = len(T2) + 1 # number of columns in G
    L_second = second_order_Tikhonov_regularization(N)

    # Lreg: column array [G  lambda*L_2] 
    # lsq_rd: column array [d  0]
    Lreg = np.vstack((G, eps*L_second)) # append the regularization matrix to G                       
    lsq_rd = np.append(resampled_data_weighted, np.zeros(N))  

    # m: solution vector to argmin(x) ||Ax-b||_2 for x >=0
    # rnorm: the residual ||Ax-b||_2     
    m, rnorm = nnls(Lreg, lsq_rd) # inversion using non-negative least-squares
    
    dsyn = np.zeros((len(resampled_time),3))
    r =np.zeros(2)
  
    dsyn[:,0] = resampled_time    # resampled time
    dsyn[:,1] = np.dot(Lwoweightreg, m) # fitted data
    dsyn[:,2] = reampled_data_nonweighted# resampled data
    #r[0] = np.linalg.norm(np.subtract(dsyn[:,2], dsyn[:,1])) # residual norm
    #r[1] = np.linalg.norm(rstdev) #data tol
    r[0] = rnorm
    r[1] = np.linalg.norm(m)# model norm
    return (m, r, dsyn)


def second_order_Tikhonov_regularization(N):
    """
    The second derivative operator L_2 of a finite-difference approximation L_2*m that is 
    proportional to the second derivative of m

    input
    ----------
    N : int
        Number of relaxation times (i.e. T2).
        
    output
    -------
    L_second : class:`numpy.ndarray`
        The second derivative operator L_2  
        
    Example:   
        array([[-2.,  1.,  0., ...,  0.,  0.,  0.],
               [ 1., -2.,  1., ...,  0.,  0.,  0.],
               [ 0.,  1., -2., ...,  0.,  0.,  0.],
               ...,
               [ 0.,  0.,  0., ..., -2.,  1.,  0.],
               [ 0.,  0.,  0., ...,  1., -2.,  1.],
               [ 0.,  0.,  0., ...,  0.,  1., -2.]]) 161*161    
    """
    L_second = (np.eye(N, N)*(-2) + np.eye(N, N, k=-1) +
                np.eye(N, N, k=1))
    return L_second

def createKernelMatrix(t, T2):
    """
    Create kernel matrix using the signal time vector't' and the relaxation time vector 'T' 

    input
    ----------
    t : 1-D array(M)
        length of the time vector
    T2 : 1-D array(N)
        length of the relaxation time vector

    output
    -------
    G :  2-D matrix (M*N)
        kernel matrix

    """    
 
    G = np.zeros((len(t),len(T2)))  
    
    tr = mat.repmat(t[:,np.newaxis], 1, len(T2))
    Tr = mat.repmat(T2[np.newaxis,:], len(t), 1) # len(t) * len(T)
    G = np.exp(-tr/Tr)
#     for i in range(len(G)):
#         G[i,:] = G[i, :]/std[i]   
    
    return G
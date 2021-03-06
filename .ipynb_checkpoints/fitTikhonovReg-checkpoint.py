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
import numpy.matlib as mat
import math 
from scipy.optimize import nnls

def T2NNLS(d,t,noise,T2,eps, winsize):
    """
    [m, dsyn, r]  = T2NNLS(d,t,noise,T2,eps)
    
    Parameters:
        d: data vector corresponding to t, NO time ZERO allowed!!
        t: time vector corresponding to d
        noise: noise vector corresponding to t and d
        T2: vector of available T2 values to which the distribution is fit
        eps: regularization factor
        winsize: size of moving average window

    Returns: 
        m: inverted model 1, length(T2)+1 with m(length(T2)+1) = baseline offset 
        dsyn: modeled data with dysn(:,1) = resampled t and dysn(:,2) = M(t)
        r: reidual norm and data tol r(1) = residual norm; r(2) = data totl

    """

    stdev = np.std(noise)
    n = 20 # down sampling
    
    ## clip decayed data by finding when moving average reaches some fraction of initial signal 
    # winsize = 100 
    filt = 1/winsize*np.ones(winsize) #create movinng average filter
    filt_d_conv = np.where(np.convolve(filt,d) < d[0]/1e5)[0] #find when mov avg < 1e-5 of d(1)
    if filt_d_conv.size > 0:
        enddecay = min(filt_d_conv) 
    else:
        enddecay = 0
    if enddecay < 1 or enddecay > len(d): enddecay = len(d)
    if enddecay <= winsize:  enddecay = len(d)
    
    d = d[0:enddecay] # trim d
    t = t[0:enddecay] # trim t
    noise = noise[0:enddecay] # trim noise
    
    if enddecay >= 1000:
        tlog_ideal = np.logspace(math.log10(t[0]), math.log10(t[len(t) - 1]), num = math.floor(len(t)/n), endpoint = True, dtype = float) #the ideal log spacing of data (small to large)
        # tlog_ideal = tlog_ideal[::-1] # reverse order so goes from small to large
        tempt = np.zeros(len(tlog_ideal))
        tempd = np.zeros(len(tlog_ideal))
        tempstdev = np.zeros(len(tlog_ideal))
        tempt[0] = t[0]
        tempd[0] = d[0]
        tempstdev[0] = stdev
        
        lastindex = 0
        for i in range(1,len(tlog_ideal)): # for each ideal time point
            cindex = np.argmin(np.abs(t - tlog_ideal[i])) # find closest datum to ideal point
            if cindex == lastindex: # if the closest point is the same as before assign NaN
                tempt[i] = np.nan
                tempd[i] = np.nan
                tempstdev[i] = np.nan
            else: # if closest point is unique
                tempt[i] = np.mean(t[lastindex + 1:cindex + 1]) # resampled binned = mean for pts between last datum and closest datum
                tempd[i] = np.mean(d[lastindex + 1:cindex + 1])
                tempstdev[i] = stdev/np.sqrt(cindex - lastindex)
            lastindex = cindex 
            
        nonnan = np.where(~np.isnan(tempt))[0] # clear out all points in resampled data which are NaN
        rt = tempt[nonnan]
        rd = tempd[nonnan]
        resampleddata = rd
        rstdev = tempstdev[nonnan]
        rd = rd/rstdev # weight the data by their standard deviation
        
    else:
        rt = t
        rstdev = stdev*np.ones(len(d))
        resampleddata = d
        rd = d/rstdev
        
    ##-------------------------- setup kernel matrix 
    tempMatrix = createKernelMatrix(rt, T2)
    G = np.ones((len(rt),len(T2)+1))
    Lwoweightreg = np.ones((len(rt),len(T2)+1))
    Lwoweightreg[:, :-1] = tempMatrix 
    G[:, :-1] = tempMatrix 
    G = G/rstdev[:, None] # or rstdev[:,np.newaxis] also works   
                       
    ##-------------------------- creat regularization Matrix 
    N = len(T2) + 1 # number of columns in G
    L_second = second_order_Tikhonov_regularization(N)

    # Lreg: column array [G  lambda*L_2] 
    # lsq_rd: column array [d  0]
    Lreg = np.vstack((G, eps*L_second)) # append the regularization matrix to G                       
    lsq_rd = np.append(rd, np.zeros(N))  
     
    m, rnorm = nnls(Lreg, lsq_rd) # inversion using non-negative least-squares
    
    dsyn = np.zeros((len(rt),3))
    r =np.zeros(2)
  
    dsyn[:,0] = rt    # resampled time
    dsyn[:,1] = np.dot(Lwoweightreg, m) # modeled data
    dsyn[:,2] = np.subtract(resampleddata, dsyn[:,1])# residuals
    
    r[0] = np.linalg.norm(dsyn[:,2]) # residual norm
   # r[1] = np.linalg.norm(rstdev) #data tol
    r[1] = np.linalg.norm(m)# model norm
    return (m,r,dsyn)


def second_order_Tikhonov_regularization(N):
    """
    The second derivative operator L_2 of a finite-difference approximation L_2*m that is 
    proportional to the second derivative of m

    Parameters
    ----------
    N : int
        Number of relaxation times (i.e. T2).
        
    Returns
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

    Parameters
    ----------
    t : 1-D array(M)
        length of the time vector
    T2 : 1-D array(N)
        length of the relaxation time vector

    Returns
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
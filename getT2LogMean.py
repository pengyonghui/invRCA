# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:51:58 2020

@author: yonghui
"""

import numpy as np

def getT2LogMean(T2, T2_dist):
    """    
    Calculates the T2 log mean value out of a relaxation time distribution
    
    Parameters
    ----------
    T2 : list array
        a list of t2 relaxation times
    T2_dist : list array
        relaxation time distribution

    Returns
    -------
    T2ml : a float value
        logarithmic mean T2

    """
    T2ml = 10**(np.sum(T2_dist*np.log10(T2))/np.sum(T2_dist))
    
    return T2ml
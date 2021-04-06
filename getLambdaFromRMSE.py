# -*- coding: utf-8 -*-

import numpy as np

def getLambdaFromRMSE(epsilon, rmse):
    """
    Input:
    epsilon: range of lambda values
    rmse: corresponding rmse values
    output:
    index: index of optimal lambda
    """
    epsilon = np.array(epsilon)
    rmse = np.array(rmse)
    # ratio between adjacent rmse values
    ratio = 1 - rmse[:-1]/rmse[1:]
    # spread of the rmse values
    span = np.around(np.max(rmse)/np.min(rmse))
    # threshold values
    threshold = span*1e-3
    # index of optimal lambda    
    index = np.where(ratio > threshold)[0][0]
    
    return index    
# -*- coding: utf-8 -*-

import numpy as np

def get_lambda_from_Lcurve(epsilon, rmse):    
	"""    
	Calculates the lambda value (smooth constraint)
	
	Parameters
	----------
	lambda/epsilon : list array
		a list of different lambda values
	rmse : list array
		corresponding rmse values

	Returns
	-------
	index : a num value
		index of optimal lambda

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
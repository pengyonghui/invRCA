# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 18:32:26 2021

@author: yonghui
"""
import matplotlib.pyplot as plt  
import math

def plt_raw_rca(arr,ax): 

	"""
	func: plot the lab measured NMR data saved in a np.array file

	input: np.array(df), where the 1-3 column is time, data, and noise;
		   ax: plot canvas

	ouput: plot canvas (ax)
	"""
	time = arr[:,0]/1000
	data = arr[:,1]
	noise = arr[:,2]    
	ax.plot(time, noise, color = 'gray', label = 'Noise')
	ax.plot(time, data,  color = 'blue', label = 'NMR signal')    

# after optimized inversion    
def plt_fitted_rca(dsyn, arr, ax):
	data = arr[:,1]
	ax.plot(dsyn[:,0], dsyn[:,1], '-', color = 'r', label='fitted')    
	ax.set_ylim(math.floor(min(data)), math.ceil(max(data)))
	ax.grid(True, which='both')
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('NMR signal amplitude')
	ax.legend(loc='best')
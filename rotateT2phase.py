# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 12:36:04 2021

@author: yonghui
"""

import numpy as np

def rotate_T2phase(arr):
	"""
	func: rotate the lab measured NMR data which was saved in a np.array

	input: NMR signal in a np.array file, where the 1-3 column is time, data, and noise

	ouput: rotated NMR signal(time, data, noise)
	"""
	print(f'the shape of the input array: {arr.shape} and the size: {arr.size}')
	#data points used to rotate the dataset. 
	n_rotate = 20

	time = arr[:,0]/1000
	data = arr[:,1]
	noise = arr[:,2]    

	x = data
	y = noise
	z = data + 1j*noise

	temp_zphase = np.angle(z)
	ztheta = np.mean(temp_zphase[0:n_rotate])
	z = z*np.exp(-1j*ztheta)
	noise = z.imag
	data = z.real 
	
	return data, time, noise

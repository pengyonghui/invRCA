# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 12:36:04 2021

@author: yonghui
"""

import pandas as pd
import numpy as np

def read_csv(csvFile):
	"""
	func: read the lab measured NMR data saved in a .csv file

	input: dir to a .csv file, where the 1-3 column is time, data, and noise

	ouput: np.array
	"""
	try: 
		df = pd.read_csv(csvFile, sep = ',', header = None)
		print("##########")
		print('importing data from:', csvFile)  
		print(',and shape of the loaded csv file as pandas.dataframe:', df.shape)

	except:
		return None

	return np.array(df)

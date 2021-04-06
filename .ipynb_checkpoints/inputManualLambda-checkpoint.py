# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:09:47 2020

@author: yonghui
"""
import os.path

def sampleLambda(fileDir, site, dartMatName):
    
    filePath = os.path.join(fileDir, site, site + '_lambdaManual.xlsx') 
    
    return sampleEps
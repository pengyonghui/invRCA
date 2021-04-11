# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt

def pltLambdaRMS(lambDa, rmse, index, ax):
    ax.semilogx(lambDa,rmse,'o-')
    ax.semilogx(lambDa[index], rmse[index],'r+')
    
    print("The chosen lambda is ", lambDa[index])

    ax.grid(True, which='both')
    # ax.plot(inv.response(), ab2, 'b-', label='fitted')
    ax.set_ylim((min(rmse), max(rmse)))
    ax.grid(True, which='both')
    ax.set_xlabel('Regularization parameter $\lambda$')
    ax.set_ylabel('Residual norm $||Gm-d||_2$')
    
def pltLcurve(rmse, modelnorm, index, ax):
    ax.plot(rmse,modelnorm,'o-')
    ax.plot(rmse[index], modelnorm[index],'r+')

    ax.grid(True, which='both')
    # ax.plot(inv.response(), ab2, 'b-', label='fitted')
    ax.set_ylim((min(modelnorm), max(modelnorm)))
    ax.set_xlim((min(rmse), max(rmse)))
    ax.grid(True, which='both')
    ax.set_xlabel('Residual norm $||Gm-d||_2$')
    ax.set_ylabel('Model norm $||m||_2$')
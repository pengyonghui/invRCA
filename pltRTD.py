# -*- coding: utf-8 -*-
import numpy as np
    
def pltT2dist(T2, m, ax):

    ax.grid(True, which='both')
    ax.set_axisbelow(True)
    ax.set_xlim(min(T2), max(T2))

    ax.bar(T2[:-2:3],m[:-2:3],width= 0.1*np.array(T2[:-2:3]), ec ='b')
    ax.plot(T2, m, 'r')
    ax.set_xscale('log')
    ax.set_xlabel('$T_2$ (s)')
    ax.set_ylabel('Signal Amplitude (a.u)')
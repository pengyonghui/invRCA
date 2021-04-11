# -*- coding: utf-8 -*-
import numpy as np

def pltT2dist_bar(T2, m, ax):
#     fig = plt.figure(figsize = (5, 4))
#     ax = fig.add_subplot(111)
    ax.grid(True, which='both')

    ax.bar(T2[:-2:3],m[:-2:3],width= 0.1*np.array(T2[:-2:3]), ec ='r')
    ax.plot(T2,m[:-1],'b')
    ax.set_xscale('log')
    ax.set_xlabel('$T_2$ (s)')
    ax.set_ylabel('Water content (%)')   

        
def pltT2dist_bothFreq(T2, m, ax, colors, depths, ith_depth, lineStyle = '-', labels = 'Dart'):
#     fig = plt.figure(figsize = (5, 4))
#     ax = fig.add_subplot(111)
    ax.grid(True, which='both')
    
    if ith_depth == 0:
        ax.semilogx(T2, m[:-1], color = colors, linestyle = lineStyle, label = labels)  
        ax.legend(loc='best')
    else:        
        ax.semilogx(T2, m[:-1], color = colors, linestyle = lineStyle)    

    ax.set_xscale('log')
    ax.set_xlim((min(T2), max(T2)))
    
    if ith_depth == len(depths)//2:
        ax.set_ylabel('Water content (%)') 

    # xticks only for the bottom axis
    if ith_depth == len(depths)-1:
        ax.set_xlabel("$T_2 (s)$", fontsize = 14,fontweight="bold")  
        ax.set_xticklabels([])  
    else:
        ax.set_xticklabels([])   
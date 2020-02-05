# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:11:06 2019

@author: Markus
"""

# code is used to print the convergence results

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# calculate the brooks and gelman convergence criteria
def Rhat(fit,plot_range,parm='gamma0',M=4):
    theta=fit.to_dataframe(pars=parm,inc_warmup=True) # write (part of fit) as panda dataframe 
    N=sum(plot_range==1)/M
    mu_parm_chain=[np.mean(theta[(theta['chain']==i) & plot_range][parm]) for i in range(M)]
    mu_parm=np.mean(mu_parm_chain)
    B_N=1/(M-1)*sum((mu_parm_chain-mu_parm)**2)
    W=1/(M*(N-1))*sum([sum((theta[(theta['chain']==i) & plot_range][parm]-mu_parm_chain[i])**2) for i in range(M)])
    tot_var=(N-1)/N*W+B_N
    Rhat=np.sqrt(tot_var/W)      
    return Rhat
    
parm="gamma0"
theta=fit.to_dataframe(pars=parm,inc_warmup=True) # write (part of fit) as panda dataframe
draw_min=-500 # 1 ist 1 draw after burn in, -500 = 1 draw of burn in period
draw_max=0
burnin=500 # number of burnin draws
plot_range=(theta['draw']<draw_max) &(theta['draw'] >=draw_min) # boolean for plotting range


fig, ax = plt.subplots() 
sns.set_style("white") # theme
Scale_Reduc_Factor=round(Rhat(fit,plot_range,parm=parm),2)

# x is the draw number
x=theta[(theta['chain']==0) & plot_range]['draw']+burnin

# y is the realization parameter draw at position x
y1=theta[(theta['chain']==0) & plot_range][parm]
y2=theta[(theta['chain']==1) & plot_range][parm]
y3=theta[(theta['chain']==2) & plot_range][parm]
y4=theta[(theta['chain']==3) & plot_range][parm]

# plot all chains
plt.plot(x,y1,'b--', label='chain 1')
plt.plot(x,y2,'g--', label='chain 2')
plt.plot(x,y3,'y--', label='chain 3')
plt.plot(x,y4,'r--', label='chain 4')

# set y axes
ax.set_ylim([-1,5])

# label and title
ax.set_ylabel('$\gamma_0$')
ax.set_title('$\widehat{R}$='+str(Scale_Reduc_Factor),fontsize=15)
ax.set_xlabel("# draw")
plt.show()  
    

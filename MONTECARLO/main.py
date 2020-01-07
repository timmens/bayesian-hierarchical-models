# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:27:43 2019

@author: Markus Schick
"""

#import sys
#sys.path.append('C:/Users/Markus/Dropbox/UNI/Research_Econometrics')

from utility import initialize_model
from linetimer import CodeTimer
from monte_carlo import monte_carlo_func

""" START MONTE CARLO"""
""" set parameter """
school_parm_alpha = {'a':0,'gamma0':0 , 'gamma1': 1, 'u':{'mu' : 0, 'sigma' :1}, 
        'eta':{'mu' : 0, 'sigma' :1}} 
school_parm_beta = {'b': 1, 'gamma0': 0, 'gamma1': 1, 'u':{'mu' : 0, 'sigma' :1}, 
        'eta':{'mu' : 0, 'sigma' :1}} 
individ_parm={'alpha' : 0,'x':{'mu':0,'sigma':1},
              'eps':{'mu':0,'sigma':1}}
parameter={'alpha': school_parm_alpha, 'beta': school_parm_beta, 'y': individ_parm}
percentile=[0.025, 0.25, 0.5, 0.75, 0.975]  

Jvec=[5, 20, 50, 100] # number of classes
Nvec=[20, 50, 100, 500] # number of pupils PER class

stan_model_1=initialize_model(model_type="rand_slope_model",prior="true")


J=Jvec[0] # J for this mc run
N=Nvec[0]

#print("J is "+str(J)+" and N is "+str(N))


mc_results=monte_carlo_func(J=J,N=N,parameter=parameter,stan_model=stan_model_1 ,model="rnd_slope",maxNsim=1)
mc_results.to_excel("runno1.xlsx")


#c_results.to_csv(index=False)




print("J is "+str(J)+" and N is "+str(N))
with CodeTimer('1 run'):
    mc_results=monte_carlo()
    mc_results=monte_carlo(J,N,parameter,rand_sl_stan,"rnd_slope",maxNsim=1)

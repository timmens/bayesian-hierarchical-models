# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:27:43 2019

@author: Markus Schick
"""
import pickle
import sys
# set path to folder with other programs (monte_carlo.py, utility.py)
sys.path.append('C:/Users/Markus/Dropbox/UNI/Research_Econometrics') 

from utility import initialize_model
from linetimer import CodeTimer
from monte_carlo import monte_carlo_func


""" START MONTE CARLO"""


""" set parameter of hierachical model"""
# random intercept model (NOT USED IN MONTE CARLO)
school_parm_alpha = {'a':0,'gamma0':1 , 'gamma1': 1, 'u':{'mu' : 0, 'sigma' :1}, 
       'eta':{'mu' : 0, 'sigma' :1}} 
# random slope
school_parm_beta = {'b': 1, 'gamma0': 1, 'gamma1': 1, 'u':{'mu' : 0, 'sigma' :3}, 
        'eta':{'mu' : 0, 'sigma' :1}} 
# individual model
individ_parm={'alpha' : 0,'x':{'mu': 0, 'sigma':3},
              'eps':{'mu':0,'sigma':1}}
# combine to one dictionary
parameter_dict={'beta': school_parm_beta, 'y': individ_parm}


""" set priors """
model_type_sim="rand_slope_model" # name of model
prior_list=["wrong","weakwrong"] # short name of prior distributions


MaxSim=2 # number of simulation runs
sample_size=[10,100] # [number of groups J, ]



for i in [0]:
    stan_model=initialize_model(model_type=model_type_sim,prior=prior_list[i]) # compile stan code into C++   
    J=sample_size[0] # number of groups J
    N=sample_size[1] # inividual per group N
    Name_of_run=model_type_sim+prior_list[i]+"J"+str(J)+"N"+str(N)+"abbrev" # build name of run
    print(Name_of_run+" with "+str(MaxSim)+" simulation runs")
    ct = CodeTimer(Name_of_run,unit='h') # stop the time!
    with ct: # monte carlo call
        mc_results,dictionary=monte_carlo_func(SimName=Name_of_run,J=J,N=N,parameter=parameter_dict,stan_model=stan_model ,model="rnd_slope",maxNsim=MaxSim)
    
    # save results in dataframe 
    
    # Store dataframe 
    mc_results.to_excel(Name_of_run+".xlsx")
    
    # Store dictionary
    dictionary['zeit']=ct.took  #contains run time (in hours)   
    with open(Name_of_run+'.pickle', 'wb') as handle: 
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL) # store object efficiently with pickle

    
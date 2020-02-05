# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:22:20 2020

@author: Markus
"""

import sys
sys.path.append('C:/Users/Markus/Dropbox/UNI/Research_Econometrics')

import pandas as pd
import pickle
from post_analysis_helper import plot_means
from post_analysis_helper import plot_posterior
from post_analysis_helper import print_kpis
from post_analysis_helper import analyze_eta


Name_of_run='rand_slope_modelwrongJ10N200'

# load dataframe
df=pd.read_excel(Name_of_run+".xlsx") 

#Load dictionary
with open(Name_of_run+'.pickle', 'rb') as handle:
    dictionary = pickle.load(handle)

# get information about run
print(dictionary['parametrization']) # parametrization
dictionary['stan_model'].show() # stan model

# convergence results
print("legitimate runs ratio")
print(sum(df['sim_flg']==1)/len(df['sim_flg']==1))
print("Average divergence in %")
print(sum(df['converge_ratio'])/len(df['sim_flg']==1)) 
print("legitimate runs ratio")



print_kpis(dictionary) # print kpi to table
plot_means(dictionary,maxNsim=500) # plot posterior means
plot_posterior(dictionary,maxNsim=500) # plot posterior
analyze_eta(dictionary) # plot eta on gamma1

















# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:57:54 2020

@author: Markus
"""

import numpy as np
import pandas as pd

#from utility import parm_as_j_list
from draw_analysis import check_eff_sample_size
from draw_analysis import check_rhat
from draw_analysis import calc_divergence_ratio
from utility import  clean_dict
from utility import draw_J_random
from utility import draw_N_random


""" data generating process """

def fixed_sample(nvec,parameter):
    # This function draws the fixed variables of our monte calro_study.
    # e.g. group and individual characteristics x and u
    
    N=sum(nvec) # total number of individals N=sum[3,3,3,3]
    J=len(nvec) # total number of schools J=len[3,3,3,3]
    school=[i for i in range(len(nvec)) for _ in range(nvec[i])]    # schoool j of every i student 
    parms =parameter['beta'] # parmaeterization for random slope group level   
    x=draw_N_random(N,mu=parameter['y']['x']['mu'],sigma=parameter['y']['x']['sigma']) # idividual characteristics
    u=draw_J_random(J,mu=parms['u']['mu'],sigma=parms['u']['sigma'],school=school) # group characteristics
    df = pd.DataFrame({'school': school,'x':x,'u':u}) # write vectors to dataframe
    return df


def random_sample(df,nvec,parameter,model="rnd_slope"):
    # This function draws the random variables of our monte calro_study
    # e.g. group and individual innovations eta, epsilon 
    # and calculates observations y based on them 
    
    N=sum(nvec) # total number of individals N=sum[3,3,3,3]
    J=len(nvec) # total number of schools J=len[3,3,3,3]
    
    df_new=df # initalize container
    school=df['school'].tolist() # collect vector with school belonging of individuals

    parms =parameter['beta'] # parmaeterization for the in level   
    eta=draw_J_random(J,mu=parms['eta']['mu'],sigma=parms['eta']['sigma'],school=school) # draw school innovations
    eps=draw_N_random(N,mu=parameter['y']['eps']['mu'],sigma=parameter['y']['eps']['sigma']) # draws individual innovations

    # combine group model
    gamma0=parms['gamma0']
    gamma1=parms['gamma1']
    beta=gamma0+gamma1*df['u']+eta
  
    # combine individual model
    alpha=parameter['y']['alpha']*np.ones(N)  # inividual interecpet
    y=alpha+beta*df['x']+eps 
    
    # to dataframe
    df_new['eta']=eta  #
    df_new['beta']=beta
    df_new['y']=y
    
    return df_new



def monte_carlo_func(SimName,J,N,parameter,stan_model,model,maxNsim=5):
    # This function performs the monte carlo procedure for some stan_model
    
    
    np.random.seed(500) # set seed 
    
    if model=="rnd_slope": 
        colnames=["simulation_name","model","simnum","sim_flg","converge_ratio","N","J"]#,"KS","MSE","Mean"]
        parm_of_int="beta"
    
    # intialize empty conatiner
    MC_summary=pd.DataFrame(columns=colnames) 
    
    # save information about the run in a dictionary
    dens_dict =	{ 
            # general information abouut run
            "Simname": SimName,
             "J": J,
             "N": N,
             "Number of simulations":maxNsim,
             "stan_model": stan_model,
             "model": model,
             "parametrization": parameter,
             
             # information of simrun i. Collected for i=1:Number of simulations
             str(parm_of_int): {"empty":0},
             "gamma0": {"empty":0},
             "gamma1": {"empty":0},
             'mu_sigma_y': {"empty":0},
             'mu_sigma_b': {"empty":0},
             'sim_flg': {"empty":0},
             'converge_rat': {"empty":0},
             "ml"+str(parm_of_int): {"empty":0},
             "mlgamma0": {"empty":0},
             "mlgamma1": {"empty":0}}
    
    nvec=[N for _ in range(J)] # write the number of individuals per school (N) for J groups
    sumN=sum(nvec) # total number of individuals
    
    df_fixed=fixed_sample(nvec,parameter) # draw random sample once
        
    dens_dict['u']=df_fixed['u'][0] # save group characteristics of group 1 in dictionary

    for nsim in range(maxNsim):   # loop over all simulation runs
        
        df=random_sample(df_fixed,nvec,parameter)  # draw random sample every time
        
        # write data in stan format
        u_schools=pd.unique(df['u']) # u_b is Nx1; write as Jx1
        sim_data= {'N': sumN,
                  'J': J,
                 'school': [i+1 for i in df['school'].tolist()], # Stan counts starting at 1
                 'u' : u_schools,
                 'x': df['x'].tolist(),
                 'y': df['y'].tolist()}
                
        # stan options 
        control={'adapt_delta':0.8, 'max_treedepth':10} # default controls
        #control={'adapt_delta':0.8, 'max_treedepth':25} # adapted in case of non-convergence, stan warnings or lack of efficiency.
                
        # draw samples from the model
        fit=stan_model.sampling(control=control,data=sim_data,iter=2000,warmup=500, chains=4,seed=500)      
        # fit object includes all parameter draws and convergence results      
      
        # check convergence, effective sample size and divergence and save in dict
        sim_flg=check_rhat(fit,quiet=True)*check_eff_sample_size(fit,quiet=True)
        converge_rat=calc_divergence_ratio(fit,quiet=True)
        dens_dict['sim_flg'][str(nsim)]=sim_flg
        dens_dict['converge_rat'][str(nsim)]=converge_rat
        
        # save draws for gamma0, gamma1 and beta_1
        dens_dict["gamma0"][str(nsim)]=fit['gamma0']
        dens_dict["gamma1"][str(nsim)]=fit['gamma1']
        dens_dict[str(parm_of_int)][str(nsim)]=fit[str(parm_of_int)+"[1]"]     
        
        # save mean estimates for standard deviations
        dens_dict['mu_sigma_y'][str(nsim)]=np.mean(fit['sigma_y'])
        dens_dict['mu_sigma_b'][str(nsim)]=np.mean(fit['sigma_b'])

        # write run characteristics to dataframe
        data=np.array([SimName,model,nsim,sim_flg,converge_rat,N,J])
        MC_summary_new=pd.DataFrame(columns=colnames, data=[data])
        MC_summary=MC_summary.append(MC_summary_new)
        
        print(str(nsim)) #feedback on the status of the simulation
        
    clean_dict(dens_dict,str(parm_of_int)) # pop intial values
    
    return MC_summary, dens_dict # return dataframe and dictionary 

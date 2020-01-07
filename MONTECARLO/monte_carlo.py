# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:57:54 2020

@author: Markus
"""

import numpy as np
import pandas as pd

from utility import parm_as_j_list
from draw_analysis import check_div
from draw_analysis import check_eff_sample_size
from draw_analysis import get_true_beta_parm
from draw_analysis import get_emp_percentiles
from draw_analysis import get_true_percentiles
from draw_analysis import calc_perc_diff
from draw_analysis import calc_MSE
from draw_analysis import calc_avg_KS


""" data generating process """
def mc_sample(nvec,parameter,model="rnd_slope"):
    N=sum(nvec)
    J=len(nvec)
    # schoool j of every i student 
    school=[i for i in range(len(nvec)) for _ in range(nvec[i])]
    
    def draw_J_random(mu=0,sigma=1,school=school):
        # draw J random characteristics for n observations
        x=np.random.normal(mu,sigma,J)
        return x[school]    
    
    def draw_N_random(mu=0,sigma=1,N=N):
        return np.random.normal(mu,sigma,N)

    def rnd_coeff(coeff): # take parametrization from dictionary
        parms =parameter[coeff]
        gamma0=parms['gamma0']
        gamma1=parms['gamma1']
        u     =draw_J_random(mu=parms['u']['mu'],sigma=parms['u']['sigma'])
        eta   =draw_J_random(mu=parms['eta']['mu'],sigma=parms['eta']['sigma'])
        rnd_coeff=gamma0+gamma1*u+eta
        return u, rnd_coeff, eta
    
    def rnd_obs(obs):
        parms =parameter[obs]
        x     =draw_N_random(mu=parms['x']['mu'],sigma=parms['x']['sigma'])
        eps   =draw_N_random(mu=parms['eps']['mu'],sigma=parms['eps']['sigma'])
        return x,eps
    
    x,eps=rnd_obs('y')
    if model=="rnd_slope":
        u_b, beta, eta_b=rnd_coeff('beta')
        alpha=parameter['alpha']['a']*np.ones(N)
        df = pd.DataFrame({'school': school,'y': alpha+beta*x+eps,'x':x,'eps':eps,'alpha':alpha,'beta':beta,'u':u_b, 'eta_b':eta_b})
    if model=="rnd_intercept":
        beta=parameter['beta']['b']*np.ones(N)
        u_a, alpha, eta_a=rnd_coeff('alpha')
        df = pd.DataFrame({'school': school,'y': alpha+beta*x+eps,'x':x,'eps':eps,'alpha':alpha,'u':u_a,'eta_a': eta_a, 'beta':beta})
    return df


# monte carlo
def monte_carlo_func(J,N,parameter,stan_model,model,maxNsim=5,percentile=[0.025 ,0.25, 0.5, 0.75, 0.975]):
    
    if model=="rnd_slope":
        colnames=["model","simnum","converge_flg","N","J","rnd parm diff","emp. gamma0 perc","emp.gamma1 perc","KS","MSE"]
        parm_of_int="beta"
    if model=="rnd_intercept":
        colnames=["model","simnum","converge_flg","N","J","rnd parm diff","emp. gamma0 perc","emp.gamma1 perc","KS","MSE"]
        parm_of_int="alpha"

    # intialize empty conatiner
    MC_summary=pd.DataFrame(columns=colnames)
    
    nvector=[N for _ in range(J)]
    sumN=sum(nvector)
    for nsim in range(maxNsim):   
        #print("start sim run no"+str(nsim))
        # draw data
        df=mc_sample(nvector,parameter,model=model)
        # write data in stan format
        u_schools=pd.unique(df['u']) # u_b is Nx1; write as Jx1
        sim_data= {'N': sumN,
                  'J': J,
                 'school': [i+1 for i in df['school'].tolist()], # Stan counts starting at 1
                 'u' : u_schools,
                 'x': df['x'].tolist(),
                 'y': df['y'].tolist()}
            
        control={'adapt_delta':0.84, 'max_treedepth':15}
        #control={'adapt_delta':0.8, 'max_treedepth':10}
        
        fit=stan_model.sampling(control=control,data=sim_data,iter=6000, chains=4) # draw samples from the model  
            
        # check convergence and effective sample size
        converge_flg=check_div(fit,quiet=True)*check_eff_sample_size(fit,quiet=True)
        
        # percentile analysis
        # theoretical percentiles of b_j/a_j
        theoretical_percentiles=get_true_percentiles(u_schools,parameter,parm_of_int,percentile)
        

        #empirical percentiles of alpha/beta, gamma1 and gamma0
        parm_name_list=parm_as_j_list(parm_of_int,J) # write beta as beta[1], beta[2], ... beta[J]
        empirical_percentiles=get_emp_percentiles(parm_name_list,fit,percentile) # beta[j] is fct of u_j
        perc_diff=calc_perc_diff(theoretical_percentiles, empirical_percentiles).tolist()
        
        gamma0_percentiles=np.quantile(fit['gamma0'],percentile).tolist()
        gamma1_percentiles=np.quantile(fit['gamma1'],percentile).tolist()
        
        # Kolmogorov–Smirnov test
        KS=calc_avg_KS(fit,u_schools,parm_of_int,J,parameter)
        
        # Mean squared error
        mu_u,_ =get_true_beta_parm(u_schools,parameter,parm_of_int) 
        tot_MSE=calc_MSE(mu_u,parm_name_list,fit)
        avg_MSE=sum(tot_MSE)/float(len(tot_MSE))

        # write results as dataframe
        data=np.array([model,nsim,converge_flg,N,J,perc_diff,gamma0_percentiles,gamma1_percentiles,KS,avg_MSE])
        MC_summary_new=pd.DataFrame(columns=colnames, data=[data])
        # append summary
        MC_summary=MC_summary.append(MC_summary_new)          
    return MC_summary


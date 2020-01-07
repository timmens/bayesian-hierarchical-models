# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:51:57 2020

@author: Markus
"""


import scipy as sp
import numpy as np

# post monte carlo analysis helpers
def get_true_beta_parm(u,parameter,parm):
        # currently just for beta
        """ extract parameter"""       
        parms =parameter[parm]
        gamma0=parms['gamma0']
        gamma1=parms['gamma1']
        mu=gamma0+gamma1*u
        sigma=parms['u']['sigma']
        return mu,sigma

def get_true_percentiles(u,parameter,parm,percentile):
    
    def get_true_beta_parm(u,parameter,parm):
        # currently just for beta
        """ extract parameter"""       
        parms =parameter[parm]
        gamma0=parms['gamma0']
        gamma1=parms['gamma1']
        mu=gamma0+gamma1*u
        sigma=parms['u']['sigma']
        return mu,sigma
    
    mu,sigma=get_true_beta_parm(u,parameter,parm)  
    true_percentiles=[sp.stats.norm.ppf(percentile, loc=mu[j], scale=sigma) for j in range(len(mu))]
    return true_percentiles


def get_emp_percentiles(var_list,fit,percentile):
    # extract stan draws for these variabels
    J=len(var_list)
    stan_draws=[fit[var] for var in var_list]
    empirical_perc=[np.quantile(stan_draws[j],percentile) for j in range(J)] #empirical
    return empirical_perc

def calc_perc_MSE(emp_percentiles,true_percentiles):
    MSE_tot=0
    for j in range(len(emp_percentiles)):
        diff=emp_percentiles[j]-true_percentiles[j]
        MSE=sum(np.square(diff))
        MSE_tot+=MSE
    return MSE_tot

def calc_MSE(mu,var_list,fit):
     J=len(var_list)
     stan_draws=[fit[var] for var in var_list]
     MSE=[sum((stan_draws[j]-mu[j])**2)/len(stan_draws[j]) for j in range(J)]   
     return MSE         
        
        
def calc_perc_diff(percentile1,percentile2):
    J1=len(percentile1) 
    J2=len(percentile2)
    if (J1-J2)==0:
        for j in range(1,J1):
             percentile1[0]+=percentile1[j]
             percentile2[0]+=percentile2[j]
        diff=percentile1[0]-percentile2[0]
        return diff
    else:   
        print("theo has len"+str(J1)+"and emp has len"+str(J2))
        return False


    
def calc_avg_KS(fit,u,parm_of_int,J,parameter):
    def parm_as_j_list(parm,J): # write beta as [beta[1] beta[2] etc.]
        return [parm+str([i+1]) for i in range(J)] #stan starts at 1
    parm_name_list=parm_as_j_list(parm_of_int,J) # write beta as beta[1], beta[2], ... beta[J]
    def get_true_beta_parm(u,parameter,parm):
        # currently just for beta
        """ extract parameter"""       
        parms =parameter[parm]
        gamma0=parms['gamma0']
        gamma1=parms['gamma1']
        mu=gamma0+gamma1*u
        sigma=parms['u']['sigma']
        return mu,sigma
    true_parms=get_true_beta_parm(u,parameter,parm_of_int)
    def KS_j(j,parm_name_list,true_parm):
        parm_name=parm_name_list[j]
        y=fit[parm_name]
        true_parameter=(true_parms[0][j],true_parms[1])
        KS_stat,KS_pval=sp.stats.kstest(y, 'norm',args=true_parameter)
        return KS_stat,KS_pval # currently only return statistic and ingore pvalue
    KS_list=[KS_j(j,parm_name_list,true_parms) for j in range(J)]
    return np.mean(KS_list)


""" convergence checks: based on package stan_utility 
https://github.com/grburgess/stan_utility """


def check_eff_sample_size(fit, quiet=False):
    """Checks the effective sample size per iteration"""
    fit_summary = fit.summary(probs=[0.5])
    n_effs = [x[4] for x in fit_summary["summary"]]
    for n_eff in n_effs:
        if n_eff < 1000:

            return False
        else:
            return True

def check_div(fit, quiet=False):
    """Check transitions that ended with a divergence"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    divergent = [x for y in sampler_params for x in y["divergent__"]]
    n = sum(divergent)
    N = len(divergent)
    if not quiet:
        print(
            "{} of {} iterations ended with a divergence ({}%)".format(
                n, N, 100 * n / N
            )
        )
    if n > 0:
        if not quiet:
            print("  Try running with larger adapt_delta to remove the divergences")
        else:
            return False
    else:
        if quiet:
            return True
        
        
def check_n_eff(fit, quiet=False):
    """Checks the effective sample size per iteration"""
    fit_summary = fit.summary(probs=[0.5])
    n_effs = [x[4] for x in fit_summary["summary"]]
    names = fit_summary["summary_rownames"]
    n_iter = len(fit.extract()["lp__"])
    no_warning = True
    for n_eff, name in zip(n_effs, names):
        ratio = n_eff / n_iter
        if ratio < 0.001:
            if not quiet:
                print("n_eff / iter for parameter {} is {}!".format(name, ratio))
            no_warning = False
    if no_warning:
        if not quiet:
            print("n_eff / iter looks reasonable for all parameters")
        else:
            return True
    else:
        if not quiet:
            print(
                "  n_eff / iter below 0.001 indicates that the effective sample size has likely been overestimated"
            )
        else:
            return False
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:27:43 2019

@author: Markus Schick
"""



import numpy as np
import pandas as pd
import random
from scipy.stats import norm




""" TO DO:
    -> differentiate between two types of errors: montecarlo error & model error
    -> find minimum # of stan draws/ burn in (for all models)
"""

""" data generating process """

def mc_sample(nvec,parameter,model="rnd_slope"):    
    N=sum(nvec)
    J=len(nvec)
    #seed=random.seed()ndom
    
    # schoool j of every i student 
    school=[i for i in range(len(nvec)) for _ in range(nvec[i])]
    
    def draw_J_random(mu=0,sigma=1,school=school):
        # draw J random characteristics for n observations
        x=np.random.normal(mu,sigma,J)
        return x[school]    
    
    def draw_N_random(mu=0,sigma=1,N=N):
        return np.random.normal(mu,sigma,N)

    def rnd_coeff(coeff,seed=random.seed()): # take parametrization from dictionary
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



# post monte carlo analysis helpers
def get_true_percentiles(u,udict,parameter,parm,J,percentile):
    def get_true_beta_parm(u,udcit,parameter,J):
        # currently just for beta
        """ extract parameter"""
        # write this function indep. of u_b or u_a        
        parms =parameter[parm]
        gamma0=parms['gamma0']
        gamma1=parms['gamma1']
        mu=gamma0+gamma1*u
        sigma=parms[udict]['sigma']
        return mu,sigma
    
    mu,sigma=get_true_beta_parm(u,udict,parameter,J)  
    true_percentiles=[norm.ppf(percentile, loc=mu[j], scale=sigma) for j in range(J)]
    return true_percentiles
    
        
def parm_as_j_list(parm,J): # write beta as [beta[1] beta[2] etc.]
    return [parm+str([i+1]) for i in range(J)] #stan starts at 1


def get_emp_percentiles(var_list,fit):
    # extract stan draws for these variabels
    stan_draws=[fit[var] for var in var_list]
    empirical_percentiles=[np.quantile(stan_draws[j],percentile) for j in range(J)] #empirical
    return empirical_percentiles

def calc_MSE(emp_percentiles,true_percentiles):
    MSE_tot=0
    for j in range(len(emp_percentiles)):
        diff=emp_percentiles[j]-true_percentiles[j]
        MSE=sum(np.square(diff))
        MSE_tot+=MSE
    return MSE_tot

""" convergence checks: based on package stan_utility 
https://github.com/grburgess/stan_utility """
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
        
# monte carlo function        

def monte_carlo(J,N,parameter,stan_model,model,maxNsim=5):
    
    if model=="rnd_slope":
        colnames=["model","simnum","converge_flg","N","J","theo. beta perc","emp. beta perc","emp. gamma0 perc","emp.gamma1 perc"]
        parm_of_int="beta"
    if model=="rnd_intercept":
        colnames=["model","simnum","converge_flg","N","J","theo. alpha perc","emp. alpha perc","emp. gamma0 perc","emp.gamma1 perc"]
        parm_of_int="alpha"

    # intialize empty conatiner
    MC_summary=pd.DataFrame(columns=colnames)
    
    nvector=[N for _ in range(J)]
    sumN=sum(nvector)
    for nsim in range(maxNsim):   
        print("start sim run no"+str(nsim))
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
            
        #control={'adapt_delta':0.84, 'max_treedepth':15}
        #control={'adapt_delta':0.8, 'max_treedepth':10}
        
        fit=stan_model.sampling(data=sim_data,iter=4000, chains=4) # draw samples from the model  
             
        # check convergence and effective sample size
        converge_flg=check_div(fit,quiet=True)*check_n_eff(fit,quiet=True)
        # theoretical percentiles of b_j/a_j
        theoparm_percentiles=get_true_percentiles(u_schools,'u',parameter,parm_of_int,J,percentile)
    
        #empirical percentiles of alpha/beta, gamma1 and gamma0
        parm_list=parm_as_j_list(parm_of_int,J) # write beta as beta[1], beta[2], ... beta[J]
        empparm_percentiles=get_emp_percentiles(parm_list,fit) # beta[j] is fct of u_j
        gamma0_percentiles=np.quantile(fit['gamma0'],percentile) 
        gamma1_percentiles=np.quantile(fit['gamma1'],percentile) 
        
        # write results as dataframe
        data=np.array([model,nsim,converge_flg,N,J,theoparm_percentiles,empparm_percentiles,gamma0_percentiles,gamma1_percentiles])
        MC_summary_new=pd.DataFrame(columns=colnames, data=[data])
        # append summary
        MC_summary=MC_summary.append(MC_summary_new)      
        
    # append dataframe
    
    return MC_summary
    


""" START MONTE CARLO"""
""" set parameter """
school_parm_alpha = {'a':0,'gamma0':0 , 'gamma1': 1, 'u':{'mu' : 0, 'sigma' :1}, 
        'eta':{'mu' : 0, 'sigma' :1}} 
school_parm_beta = {'b': 1, 'gamma0': 0, 'gamma1': 1, 'u':{'mu' : 0, 'sigma' :1}, 
        'eta':{'mu' : 0, 'sigma' :1}} 
individ_parm={'alpha' : 0,'x':{'mu':0,'sigma':1},
              'eps':{'mu':0,'sigma':1}}
parameter={'alpha': school_parm_alpha, 'beta': school_parm_beta, 'y': individ_parm}


percentile=[0.025 ,0.25, 0.5, 0.75, 0.975]  
Jvec=[5,20,50] # number of classes
Nvec=[20,50,100,500] # number of pupils PER class
nsim=500   # monte carlo sim number


J=Jvec[0] # J for this mc run
N=Nvec[0] # class size for this mc

mc_results=monte_carlo(J,N,parameter,rand_int_stan,"rnd_intercept",maxNsim=5)




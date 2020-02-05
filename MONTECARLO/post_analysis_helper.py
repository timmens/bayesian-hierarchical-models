# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:33:19 2020

@author: Markus
"""

# plot helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def plot_means(dictionary,maxNsim=50):
    # an density estimate for gamma0 and gamma1 is plotted
    
    maxNsim=min(dictionary["Number of simulations"],maxNsim)
    
    parm_name="gamma0"
    mu_parm=[np.mean(dictionary[parm_name][str(i)]) for i in range(maxNsim)]  
    mu_parm_mean=np.mean(mu_parm)
    fig, ax = plt.subplots()
    sns.set_style("white")
    sns.kdeplot(mu_parm, shade=False, color="b")
    ax.set_ylabel('Density')
    #ax.set_xlim([-1,3])
    #ax.set_ylim([0,3])
    ax.set_xlabel("$\gamma_0")
    plt.axvline(linewidth=2,x=1)
    plt.axvline(linewidth=2,x=mu_parm_mean,linestyle='--')  
    plt.show()    

    parm_name="gamma1"
    mu_parm=[np.mean(dictionary[parm_name][str(i)]) for i in range(maxNsim)]  
    mu_parm_mean=np.mean(mu_parm)
    fig, ax = plt.subplots()
    sns.set_style("white")
    sns.kdeplot(mu_parm, shade=False, color="b")
    ax.set_ylabel('Density')
    #ax.set_xlim([-1,3])
    #ax.set_ylim([0,3])
    ax.set_xlabel("$\gamma_1")
    plt.axvline(linewidth=2,x=1)
    plt.axvline(linewidth=2,x=mu_parm_mean,linestyle='--')  
    plt.show()    

 


def plot_posterior(dictionary,maxNsim=50):
    # plot the fitted densites for the posterior draws of gamma0 and gamma1
    maxNsim=min(dictionary["Number of simulations"],maxNsim)
    
    parm_name="gamma0"
    fig, ax = plt.subplots()
    sns.set_style("white")
    mu_parms=[]
    for i in range(maxNsim):
        sns.kdeplot(dictionary[str(parm_name)][str(i)], shade=False,alpha=0.2)#,saturation=1)   
        mu_parms.append(dictionary[parm_name][str(i)])
    mu_parm_mean=np.mean(mu_parms)
    ax.set_ylabel('Density')
    ax.set_xlim([-1,3])
    ax.set_ylim([0,3])
    ax.set_xlabel("$\gamma_0$")
    plt.axvline(linewidth=2,x=1)
    plt.axvline(linewidth=2,x=mu_parm_mean,linestyle='--')
    plt.show()  

    parm_name="gamma1"  
    fig, ax = plt.subplots()
    sns.set_style("white")
    mu_parms=[]
    for i in range(maxNsim):
        sns.kdeplot(dictionary[str(parm_name)][str(i)], shade=False,alpha=0.2)#,saturation=1)   
        mu_parms.append(dictionary[parm_name][str(i)])
    mu_parm_mean=np.mean(mu_parms)
    ax.set_ylabel('Density')
    ax.set_xlim([-1,3])
    ax.set_ylim([0,8])
    ax.set_xlabel("$\gamma_1$")
    plt.axvline(linewidth=2,x=1)
    plt.axvline(linewidth=2,x=mu_parm_mean,linestyle='--')
    plt.show()      
    

    
def print_kpis(dictionary):
    # function prints the simulation run results for our appendix
    N=dictionary['N']
    J=dictionary['J']
    maxNsim=dictionary["Number of simulations"] 


    # true values
    dictionaryp=dictionary['parametrization']
    print(dictionaryp)
    gamma0_true= dictionaryp['beta']['gamma0']
    gamma1_true= dictionaryp['beta']['gamma1']
    print("true values")
    print(["gamma0","gamma1","sigma_b","sigma_y"])
    analysis0=np.array([dictionaryp['beta']['gamma0'], dictionaryp['beta']['gamma1'], dictionaryp['beta']['eta']['sigma'], dictionaryp['y']['eps']['sigma']])
    print(analysis0)

    
    # bias analysis
    mu_gamma0_s=[np.mean(dictionary['gamma0'][str(i)]) for i in range(maxNsim)]  
    mu_gamma0=round(np.mean(mu_gamma0_s),4)
    mu_gamma1_s=[np.mean(dictionary['gamma1'][str(i)]) for i in range(maxNsim)]  
    mu_gamma1=round(np.mean(mu_gamma1_s),4)
    print("bias_analysis")
    print(["J","N","gamma0","gamma1","sigma_b","sigma_y"])
    
    # special case for ealier runs
    try: # try except statement: Prior monte carlo runs did not save all standard deviations, but just the mean
         print(str(J)+"  &  "+str(N)+"  &  "+ str(mu_gamma1)+"  &  "+str(mu_gamma0)+"  &  "+str( round(dictionary['mu_sigma_b'],4))+ "  &  "+str(round(dictionary['mu_sigma_y'],4)))
    except:
            mu_sigma_b_s=[np.mean(dictionary['mu_sigma_b'][str(i)]) for i in range(maxNsim)]  
            mu_sigma_b=round(np.mean(mu_sigma_b_s),4)
            mu_sigma_y_s=[np.mean(dictionary['mu_sigma_y'][str(i)]) for i in range(maxNsim)]  
            mu_sigma_y=round(np.mean(mu_sigma_y_s),4)
            print(str(J)+"  &  "+str(N)+"  &  "+ str(mu_gamma1)+"  &  "+str(mu_gamma0)+"  &  " +str(mu_sigma_b)+ "  &  "+str(mu_sigma_y))

    # mean 
    print("[0.05,0.95] quantile of bayesian estimates of gamma0 and gamma1")
    mu_gamma0_quant0=np.round(np.quantile(mu_gamma0_s,[0.05, 0.95]),4)[0]
    mu_gamma0_quant1=np.round(np.quantile(mu_gamma0_s,[0.05, 0.95]),4)[1]
    mu_gamma1_quant0=np.round(np.quantile(mu_gamma1_s,[0.05, 0.95]),4)[0]    
    mu_gamma1_quant1=np.round(np.quantile(mu_gamma1_s,[0.05, 0.95]),4)[1] 
    print(str(J)+"  &  "+str(N)+"  &  "+"["+str("{:.2f}".format(mu_gamma0_quant0))+","+str("{:.2f}".format(mu_gamma0_quant1))+"]"+"  &  "+"["+str("{:.2f}".format(mu_gamma1_quant0))+","+str("{:.2f}".format(mu_gamma1_quant1))+"]")
 
    # percentiles
    gamma0_95_s=[np.quantile(dictionary['gamma0'][str(i)],0.95) for i in range(maxNsim)]
    gamma0_05_s=[np.quantile(dictionary['gamma0'][str(i)],0.05) for i in range(maxNsim)]
    gamma0_95=round(np.mean(gamma0_95_s),2)
    gamma0_05=round(np.mean(gamma0_05_s),2)
    gamma1_95_s=[np.quantile(dictionary['gamma1'][str(i)],0.95) for i in range(maxNsim)]
    gamma1_05_s=[np.quantile(dictionary['gamma1'][str(i)],0.05) for i in range(maxNsim)]
    gamma1_95=round(np.mean(gamma1_95_s),2)
    gamma1_05=round(np.mean(gamma1_05_s),2)
    print("coverage analysis")
    print(["J","N","gamma0","gamma1"])
    print(str(J)+"  &  "+str(N)+"  &  "+"["+str("{:.2f}".format(gamma0_05))+","+str("{:.2f}".format(gamma0_95))+"]"+"  &  "+"["+str("{:.2f}".format(gamma1_05))+","+str("{:.2f}".format(gamma1_95))+"]" +"\\")

    # coverage ratio
    print("Coverage ratio (true values inside [0.025,0.975] quantile)")
    gamma0_975_s=[np.quantile(dictionary['gamma0'][str(i)],0.975) for i in range(maxNsim)]
    gamma0_025_s=[np.quantile(dictionary['gamma0'][str(i)],0.025) for i in range(maxNsim)]
    gamma1_975_s=[np.quantile(dictionary['gamma1'][str(i)],0.975) for i in range(maxNsim)]
    gamma1_025_s=[np.quantile(dictionary['gamma1'][str(i)],0.025) for i in range(maxNsim)]
    coverage_gamma0=round(sum([gamma0_975_s[i]>gamma0_true>gamma0_025_s[i]  for i in range(maxNsim)])/maxNsim,4)
    coverage_gamma1=round(sum([gamma1_975_s[i]>gamma1_true>gamma1_025_s[i]  for i in range(maxNsim)])/maxNsim,4)
    print(["J","N","coverage gamma0","coverage gamma1"])
    print(str(J)+"  &  "+str(N)+"  &  "+str(coverage_gamma0)+"  &  "+str(coverage_gamma1))

    # median
    print("[0.05,0.95] quantile median! of bayesian estimates of gamma0 and gamma1")
    med_gamma0_s=[np.median(dictionary['gamma0'][str(i)]) for i in range(maxNsim)]  
    med_gamma1_s=[np.median(dictionary['gamma1'][str(i)]) for i in range(maxNsim)]  
    med_gamma0_quant0=np.round(np.quantile(med_gamma0_s,[0.05, 0.95]),4)[0]
    med_gamma0_quant1=np.round(np.quantile(med_gamma0_s,[0.05, 0.95]),4)[1]
    med_gamma1_quant0=np.round(np.quantile(med_gamma1_s,[0.05, 0.95]),4)[0]
    med_gamma1_quant1=np.round(np.quantile(med_gamma1_s,[0.05, 0.95]),4)[1]
    print(str(J)+"  &  "+str(N)+"  &  "+"["+str("{:.2f}".format(med_gamma0_quant0))+","+str("{:.2f}".format(med_gamma0_quant1))+"]"+"  &  "+"["+str("{:.2f}".format(med_gamma1_quant0))+","+str("{:.2f}".format(med_gamma1_quant1))+"]")
 





def analyze_eta(dictionary):
    # print joint density of eta and gamma1/gamma0
    maxNsim=dictionary["Number of simulations"]

    mu_gamma0_s=[np.mean(dictionary['gamma0'][str(i)]) for i in range(maxNsim)]  
    mu_gamma1_s=[np.mean(dictionary['gamma1'][str(i)]) for i in range(maxNsim)]      
    mu_beta_s=[np.mean(dictionary['beta'][str(i)]) for i in range(maxNsim)]  
    mu_eta_s=[mu_beta_s[i]-mu_gamma0_s[i]-mu_gamma1_s[i]*dictionary['u'] for i in range(maxNsim)]    

    df3=pd.DataFrame({'$\gamma_1$': mu_gamma1_s, '$\eta$': mu_eta_s}) 
    plot=sns.set(style="white")
    plot=sns.jointplot(x='$\gamma_1$', y='$\eta$', data=df3, kind="kde")

    plot.ax_joint.plot([1],[0],'ro')
    plot.ax_marg_y.set_ylim(-3,2.5)
    plot.ax_marg_x.set_xlim(0.7,1.35)
    plt.show()
    
    df2=pd.DataFrame({'$\gamma_0$': mu_gamma0_s, '$\eta$': mu_eta_s}) 
    plot=sns.set(style="white")
    plot=sns.jointplot(x='$\gamma_0$', y='$\eta$', data=df2, kind="kde")
    plot.ax_joint.plot([1],[0],'ro')
    plot.ax_marg_x.set_xlim(0.5,2 )
    plt.show()
    

    
def print_dens2d(dictionary):     
    # print 2 densities agains each other. calculate the correlation between estimates
    maxNsim=dictionary["Number of simulations"]
    gamma0true=dictionary['parametrization']['beta']['gamma0']
    gamma1true=dictionary['parametrization']['beta']['gamma1']
    
    mu_gamma0_s=[np.mean(dictionary['gamma0'][str(i)]) for i in range(maxNsim)]  
    mu_gamma1_s=[np.mean(dictionary['gamma1'][str(i)]) for i in range(maxNsim)]  

    df=pd.DataFrame({'$\gamma_0$': mu_gamma0_s, '$\gamma_1$': mu_gamma1_s})

    sns.set(style="white")
    sns.jointplot(x='$\gamma_0$', y='$\gamma_1$', data=df, kind="kde")
    a=sns.jointplot(x='$\gamma_0$', y='$\gamma_1$', data=df, kind="kde")
    a.ax_joint.plot([gamma0true],[gamma1true],'ro')
    x0, x1 = a.ax_joint.get_xlim()
    y0, y1 = a.ax_joint.get_ylim()
    x0=x0+0.4
    x1=x1-0.4
    rho=np.corrcoef(mu_gamma0_s,mu_gamma1_s)[1][0] # take non-diagonal element
    y0=(x0-gamma0true)*rho+gamma1true # m=delta_y/delta_x <=> delta_y=m*delta_x
    y1=(x1-gamma0true)*rho+gamma1true # m=delta_y/delta_x <=> delta_y=m*delta_x
    a.ax_joint.plot([x0,x1],[y0,y1], ':k')    
    

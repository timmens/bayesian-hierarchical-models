# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:43:55 2020

@author: Markus
"""
import pystan
import numpy as np

def initialize_model(model_type="rand_slope_model",prior="true"):
    # function to load stan model text from file
    
    # random slope model
    if model_type=="rand_slope_model":    
        # various prior distributions
        if prior=="true":
            from model_codes import rand_slope_model_true as model_string
        if prior=="true2":
            from model_codes import rand_slope_model_true2 as model_string
        if prior=="uni":
            from model_codes import rand_slope_model_uni as model_string
        if prior=="wrong":
            from model_codes import rand_slope_model_wrong as model_string
        if prior=="weakwrong":
             from model_codes import rand_slope_model_weakwrong as model_string
        if prior=="slightlywrong":   
             from model_codes import rand_slope_model_slightlywrong as model_string
        if prior=="slightlywrong2":   
             from model_codes import rand_slope_model_slightlywrong2 as model_string
        if prior=="weakslightlywrong":
             from  model_codes import rand_slope_model_weakslightlywrong as model_string
        if prior=="weakslightlywrong2":
             from  model_codes import rand_slope_model_weakslightlywrong2 as model_string
    # decomissioned
    #if model_type=="rand_intercept_model":
        #if prior=="true":
        #    from model_codes import rand_intercept_model_true as model_string
        #if prior=="uni":
        #    from model_codes import rand_intercept_model_uni as model_string
        #if prior=="wrong":
        #    from model_codes import rand_intercept_model_wrong as model_string            
    
    # compile code with pystan to c++
    stan_model=pystan.StanModel(model_code=model_string)
    
    return stan_model # return model object! Use stan_model.show() for visualising text

        
        
""" Monte Carlo utility function"""
        
def parm_as_j_list(parm,J): # write parameter as [parameter[1] parameter[2] etc.]
    return [parm+str([i+1]) for i in range(J)] #stan starts at 1

def draw_J_random(J,mu=0,sigma=1,school=[1,1,2,2]):
    # draw J random characteristics for n observations
    # school is a vector with size N with j element [1,J]
    x=np.random.normal(mu,sigma,J)
    return x[school]    
    
def draw_N_random(N,mu=0,sigma=1):
    # draw N random characteristics with n observations
    return np.random.normal(mu,sigma,N) 


def clean_dict(dictionary,parm):
        # drop the intial entry of the model
        dictionary[parm].pop('empty')
        dictionary['gamma0'].pop('empty')
        dictionary['gamma1'].pop('empty')   
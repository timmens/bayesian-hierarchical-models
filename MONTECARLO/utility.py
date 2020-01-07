# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:43:55 2020

@author: Markus
"""
import pystan

def initialize_model(model_type="rand_slope_model",prior="true"):
    if model_type=="rand_slope_model":
        if prior=="true":
            from model_codes import rand_slope_model_true as model_string
        if prior=="uni":
            from model_codes import rand_slope_model_uni as model_string
        if prior=="wrong":
            from model_codes import rand_slope_model_wrong as model_string
    if model_type=="rand_intercept_model":
        if prior=="true":
            from model_codes import rand_intercept_model_true as model_string
        if prior=="uni":
            from model_codes import rand_intercept_model_uni as model_string
        if prior=="wrong":
            from model_codes import rand_intercept_model_wrong as model_string            
    stan_model=pystan.StanModel(model_code=model_string)
    return stan_model


def parm_as_j_list(parm,J): # write beta as [beta[1] beta[2] etc.]
    return [parm+str([i+1]) for i in range(J)] #stan starts at 1

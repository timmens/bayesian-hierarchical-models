# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:02:03 2019

@author: Markus
"""


""" random slope models"""


import pystan

""" version new"""

rnd_slope_model = """
        data {
          int<lower=0> J; 
          int<lower=0> N; 
          int<lower=1,upper=J> school[N];
          vector[J] u;
          vector[N] x;
          vector[N] y;
        } 
        parameters {
          real gamma1;
          real a;
          real gamma0;
          real<lower=0,upper=100> sigma_b;
          real<lower=0,upper=100> sigma_y;
          vector[J] eta_b;
        } 
        transformed parameters {
          vector[J] beta;
          beta=u*gamma1+eta_b;
        }
        model {
          vector[N] y_hat;
          for (i in 1:N){
                y_hat[i] = a+ x[i] * beta[school[i]];
                }
          y ~ normal(y_hat, sigma_y);
          gamma1 ~ normal(1, 1); 
          gamma0 ~ normal(0, 1); 
          eta_b ~ normal(gamma0, sigma_b);          
        }
        """      

# INTIALIZE MODEL ONCE
rand_sl_stan = pystan.StanModel(model_code=rnd_slope_model)


""" TO DO: Test this model"""
rnd_intercept_model = """
        data {
          int<lower=0> J; 
          int<lower=0> N; 
          int<lower=1,upper=J> school[N];
          vector[J] u;
          vector[N] x;
          vector[N] y;
        } 
        parameters {
          real gamma1;
          real b;
          real gamma0;
          real<lower=0,upper=100> sigma_b;
          real<lower=0,upper=100> sigma_y;
          vector[J] eta_a;
        } 
        transformed parameters {
          vector[J] alpha;
          for (j in 1:J) {
            alpha[j] <-  u[j] * gamma1 + eta_a[j];
          }
        }
        model {
          vector[N] y_hat;
          for (i in 1:N){
                y_hat[i] = alpha[school[i]]+b*x[i];
                }
          y ~ normal(y_hat, sigma_y);
          gamma1 ~ normal(1, 1); 
          gamma0 ~ normal(0, 1); 
          eta_a ~ normal(gamma0, sigma_b);          
        }
        """    
    
# INTIALIZE MODEL ONCE
rand_int_stan = pystan.StanModel(model_code=rnd_intercept_model)



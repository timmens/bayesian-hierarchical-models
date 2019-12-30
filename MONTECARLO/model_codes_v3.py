# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:02:03 2019

@author: Markus
"""

import pystan



""" random slope models"""


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

""" uniform prior"""
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
          eta_b ~ normal(gamma0, sigma_b);          
        }
        """      

# INTIALIZE MODEL ONCE
rand_sl_stan_uprior = pystan.StanModel(model_code=rnd_slope_model)


""" wrong prior (one sd)"""
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
          gamma1 ~ normal(2, 1); // prior mean 1 sd from true mean
          gamma0 ~ normal(1, 1); 
          eta_b ~ normal(gamma0, sigma_b);          
        }
        """      

# INTIALIZE MODEL ONCE
rand_sl_stan_wprior = pystan.StanModel(model_code=rnd_slope_model)

""" wrong prior (2 sd)"""
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
          gamma1 ~ normal(3, 1); // prior mean 1 sd from true mean
          gamma0 ~ normal(2, 1); 
          eta_b ~ normal(gamma0, sigma_b);          
        }
        """      

# INTIALIZE MODEL ONCE
rand_sl_stan_wprior2 = pystan.StanModel(model_code=rnd_slope_model)

""" RANDOM INTERCEPT MODEL"""

""" the paralell model  y_hat= alpha[school]+b*x performs 50% worse 
compared to a simple loop
no signifant differences between 
paralell alpha <-  u* gamma1 + eta_a and loop"""


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
          alpha <-  u* gamma1 + eta_a;
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

""" uninformative priors"""
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
          alpha <-  u* gamma1 + eta_a;
          }
        model {
          vector[N] y_hat;
          for (i in 1:N){
                y_hat[i] = alpha[school[i]]+b*x[i];
                }
          y ~ normal(y_hat, sigma_y);      
        }
        """    
    
# INTIALIZE MODEL ONCE
rand_int_stan_uprior = pystan.StanModel(model_code=rnd_intercept_model)


""" wrong priors (1 sd)"""
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
          alpha <-  u* gamma1 + eta_a;
          }
        model {
          vector[N] y_hat;
          for (i in 1:N){
                y_hat[i] = alpha[school[i]]+b*x[i];
                }
          y ~ normal(y_hat, sigma_y);      
          gamma1 ~ normal(2, 1); // prior mean 1 sd from true mean
          gamma0 ~ normal(1, 1); 
          eta_a ~ normal(gamma0, sigma_b);     
        }
        """    
    
# INTIALIZE MODEL ONCE
rand_int_stan_wprior = pystan.StanModel(model_code=rnd_intercept_model)


""" wrong priors (2 sd)"""
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
          alpha <-  u* gamma1 + eta_a;
          }
        model {
          vector[N] y_hat;
          for (i in 1:N){
                y_hat[i] = alpha[school[i]]+b*x[i];
                }
          y ~ normal(y_hat, sigma_y);      
          gamma1 ~ normal(2, 1); // prior mean 1 sd from true mean
          gamma0 ~ normal(1, 1); 
          eta_a ~ normal(gamma0, sigma_b);     
        }
        """    
    
# INTIALIZE MODEL ONCE
rand_int_stan_wprior = pystan.StanModel(model_code=rnd_intercept_model)
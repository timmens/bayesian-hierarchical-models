# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:02:03 2019

@author: Markus
"""


""" random slope models"""
rand_slope_model_true = """
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
          real<lower=0> sigma_b;
          real<lower=0> sigma_y;
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


""" uniform prior"""
rand_slope_model_uni = """
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
          real<lower=0> sigma_b;
          real<lower=0> sigma_y;
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



""" wrong prior (one sd)"""
rand_slope_model_wrong = """
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
          real<lower=0> sigma_b;
          real<lower=0> sigma_y;
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


""" RANDOM INTERCEPT MODEL"""

""" the paralell model  y_hat= alpha[school]+b*x performs 50% worse 
compared to a simple loop
no signifant differences between 
paralell alpha <-  u* gamma1 + eta_a and loop"""


rand_intercept_model_true = """
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
          real<lower=0> sigma_b;
          real<lower=0> sigma_y;
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
    

""" uninformative priors"""
rand_intercept_model_uni = """
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
          real<lower=0> sigma_b;
          real<lower=0> sigma_y;
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
    

""" wrong priors (1 sd)"""

rand_intercept_model_wrong = """
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
          real<lower=0> sigma_b;
          real<lower=0> sigma_y;
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


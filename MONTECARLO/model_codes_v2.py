# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:02:03 2019

@author: Markus
"""


""" random slope models"""




""" version new"""

rnd_slope_model = """
        data {
          int<lower=0> J; 
          int<lower=0> N; 
          int<lower=1,upper=J> school[N];
          vector[J] u_b;
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
          beta=u_b*gamma1+eta_b;
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
sm = pystan.StanModel(model_code=rnd_slope_model)


""" version old"""

rnd_slope_model = """
        data {
          int<lower=0> J; 
          int<lower=0> N; 
          int<lower=1,upper=J> school[N];
          vector[J] u_b;
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
          for (j in 1:J) {
            beta[j] <-  u_b[j] * gamma1 + eta_b[j];
          }
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
sm = pystan.StanModel(model_code=rnd_slope_model)


""" TO DO: Test this model"""
rnd_intercept_model = """
        data {
          int<lower=0> J; 
          int<lower=0> N; 
          int<lower=1,upper=J> school[N];
          vector[J] u_a;
          vector[N] x;
          vector[N] y;
        } 
        parameters {
          real gamma1;
          real b;
          real<lower=0,upper=100> sigma_b;
          real<lower=0,upper=100> sigma_y;
          vector[J] eta_a;
        } 
        transformed parameters {
          vector[J] alpha;
          for (j in 1:J) {
            alpha[j] <-  u_a[j] * gamma1 + eta_a[j];
          }
        }
        model {
          vector[N] y_hat;
          for (i in 1:N){
                y_hat[i] = alpha[school[i]]+b*x[i];
                }
          y ~ normal(y_hat, sigma_y);
          gamma1 ~ normal(1, 1); 
          eta_a ~ normal(0, sigma_b);          
        }
        """    
    
# INTIALIZE MODEL ONCE
sm = pystan.StanModel(model_code=m_rnd_slope)


""" random slope and intercept model"""
rnd_slope_intercept_model = """
        data {
          int<lower=0> J; 
          int<lower=0> N; 
          int<lower=1,upper=J> school[N];
          vector[J] u_a;
          vector[J] u_b;
          vector[N] x;
          vector[N] y;
        } 
        parameters {
          vector[2] gamma1;
          real mu_a;
          real mu_b;
          real<lower=0,upper=100> sigma_a;
          real<lower=0,upper=100> sigma_b;
          real<lower=0,upper=100> sigma_y;
          vector[J] eta_a;
          vector[J] eta_b;
        } 
        transformed parameters {
          vector[J] alpha;
          vector[J] beta;
          for (j in 1:J) {
            alpha[j] <-  u_a[j] * gamma1[1] + eta_a[j];
            beta[j] <-  u_b[j] * gamma1[2] + eta_b[j];
          }
        }
        model {
          vector[N] y_hat;
          for (i in 1:N){
                y_hat[i] = alpha[school[i]]+beta[school[i]]*x[i];
                }
          y ~ normal(y_hat, sigma_y);
          gamma1 ~ normal(1, 1); 
          mu_a ~ normal(0, 1); 
          eta_a ~ normal(mu_a, sigma_a);      
          mu_b ~ normal(0, 1); 
          eta_b ~ normal(mu_b, sigma_b);   
        }
        """    
    
# INTIALIZE MODEL ONCE
sm3 = pystan.StanModel(model_code=rnd_slope_intercept_model)

          #gamma1 ~ normal(1, 1); 
# -*- coding: utf-8 -*-
"""0
Created on Thu Dec 19 15:02:03 2019

@author: Markus
"""

#  this file includes all the stan model codes we used in our analysis


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
          gamma0 ~ normal(1, 1); 
          eta_b ~ normal(gamma0, sigma_b);
          sigma_y ~ cauchy(0, 5);
          sigma_b ~ cauchy(0, 5);
        }
        """      
 


""" wrong prior"""
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
          eta_b ~ normal(gamma0, sigma_b);  
          sigma_y ~ cauchy(0, 5);
          sigma_b ~ cauchy(0, 5);
          gamma1 ~ normal(2, 1); // prior mean 1 sd from true mean
          gamma0 ~ normal(2, 1);
        }
        """      


""" uni prior"""
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
          sigma_y ~ cauchy(0, 5);
          sigma_b ~ cauchy(0, 5);          
        }
        """      

""" weak wrong prior"""
rand_slope_model_weakwrong = """
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
          gamma1 ~ normal(2, 3); // prior mean 1 sd from true mean
          gamma0 ~ normal(2, 3);
          eta_b ~ normal(gamma0, sigma_b);
          sigma_y ~ cauchy(0, 5);
          sigma_b ~ cauchy(0, 5);          
        }
        """  

""" slightly wrong prior (one sd)"""
rand_slope_model_slightlywrong = """
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
          gamma1 ~ normal(1.2, 1); // prior mean 1 sd from true mean
          gamma0 ~ normal(1.2, 1);
          eta_b ~ normal(gamma0, sigma_b);
          sigma_y ~ cauchy(0, 5);
          sigma_b ~ cauchy(0, 5);          
        }
        """ 

""" slightly wrong prior (one sd)"""
rand_slope_model_slightlywrong2 = """
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
          gamma1 ~ normal(1.5, 1); // prior mean 1 sd from true mean
          gamma0 ~ normal(1.5, 1);
          eta_b ~ normal(gamma0, sigma_b);
          sigma_y ~ cauchy(0, 5);
          sigma_b ~ cauchy(0, 5);          
        }
        """ 
        
""" slightly wrong weak prior (one sd)"""
rand_slope_model_weakslightlywrong = """
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
          gamma1 ~ normal(1.2, 3); // prior mean 1 sd from true mean
          gamma0 ~ normal(1.2, 3);
          eta_b ~ normal(gamma0, sigma_b);
          sigma_y ~ cauchy(0, 5);
          sigma_b ~ cauchy(0, 5);          
        }
        """ 
""" slightly wrong weak prior (one sd)"""
rand_slope_model_weakslightlywrong2 = """
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
          gamma1 ~ normal(1.5, 3); // prior mean 1 sd from true mean
          gamma0 ~ normal(1.5, 3);
          eta_b ~ normal(gamma0, sigma_b);
          sigma_y ~ cauchy(0, 5);
          sigma_b ~ cauchy(0, 5);          
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
          real<lower=0> sigma_a;
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
          eta_a ~ normal(gamma0, sigma_a);
          sigma_y ~ cauchy(0, 5);
          sigma_a ~ cauchy(0, 5);          
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
          real<lower=0> sigma_a;
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
          sigma_y ~ cauchy(0, 5);
          sigma_a ~ cauchy(0, 5);
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
          real<lower=0> sigma_a;
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
          eta_a ~ normal(gamma0, sigma_a);     
        }
        """    


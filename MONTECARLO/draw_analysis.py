# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:51:57 2020

@author: Markus
"""




""" convergence checks: based on package stan_utility 
https://github.com/grburgess/stan_utility """


def check_eff_sample_size(fit, quiet=False):
    """Checks the effective sample size per iteration"""
    fit_summary = fit.summary(probs=[0.5])
    n_effs = [x[4] for x in fit_summary["summary"]]
    for n_eff in n_effs:
        if n_eff < 500:
            return False
        else:
            return True

def calc_divergence_ratio(fit, quiet=False):
    """Check transitions that ended with a divergence"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    divergent = [x for y in sampler_params for x in y["divergent__"]]
    n = sum(divergent)
    N = len(divergent)
    divergence_ratio=100*n/N
    if not quiet:
        print(
            "{} of {} iterations ended with a divergence ({}%)".format(
                n, N, 100 * n / N
            )
        )        
    return divergence_ratio   


def check_rhat(fit, quiet=False):
    """Checks the potential scale reduction factors"""
    from math import isnan
    from math import isinf
    fit_summary = fit.summary(probs=[0.5])
    rhats = [x[5] for x in fit_summary["summary"]]

    for rhat in rhats:
        if rhat > 1.1 or isnan(rhat) or isinf(rhat):
            return False
        else: 
            return True
        
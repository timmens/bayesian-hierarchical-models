import numpy as np
from numpy import pi
from scipy.special import gamma
from scipy.stats import norm, multivariate_normal

def volume_ball(d):
    """Returns volume of unit ball (wrt l2 norm) in d-dim euclidian space."""
    return (pi ** (d / 2.)) / gamma(1 + d/2.)

def volume_cube(d):
    """Returns volume of unit ball (wrt sup norm) in d-dim euclidian space."""
    return 2 ** d


dimensions = [1, 2, 3, 5, 7, 10, 15]

vol_ball = [volume_ball(d) for d in dimensions]
vol_cube = [volume_cube(d) for d in dimensions]
vol_ratio = [np.round(x / y, decimals=5) for x, y in zip(vol_ball, vol_cube)]


def probability_cube(d):
    """Returns probability of a d-dim. gaussian falling in a unit ball wrt sup norm."""
    return (norm.cdf(1) - norm.cdf(-1))**d

def upperbound_probability_ball(d):
    """Returns simple upper bound on the probability of a d-dim. gaussian falling in a unit ball wrt l2 norm."""
    return multivariate_normal(np.zeros(d), np.eye(d)).pdf(np.zeros(d)) * volume_ball(d)


upper_prob_ball = [upperbound_probability_ball(d) for d in dimensions]
prob_cube = [probability_cube(d) for d in dimensions]
prob_ratio = [np.round(x/y, decimals=5) for x, y in zip(upper_prob_ball, prob_cube)]


print(f"dimensions: {dimensions}")
print(f"vol-ratio: {vol_ratio}")
print(f"prob-ratio: {prob_ratio}")

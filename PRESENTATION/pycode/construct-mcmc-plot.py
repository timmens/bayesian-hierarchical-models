import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from matplotlib import style
from scipy.stats import multivariate_normal

sns.set_style('whitegrid')

# Metropolis - (Hastings) - Algorithm
def pgauss(x, y, mean, cov):
    """Returns density of bivariate normal with mean=mean and cov=cov."""
    return st.multivariate_normal.pdf([x, y], mean=mean, cov=cov)

def metropolis_hastings(p, nsamples):
    """Returns nsamples of density p using a gaussian proposal and the metropolis algorithm."""
    x, y = -1., -1. # starting values
    samples = np.zeros((nsamples+1, 2))
    samples[0, :] = np.array([x, y])
    
    for t in range(0, nsamples):
        x_proposal, y_proposal = np.array([x, y]) + np.random.normal(size=2)
        u = np.random.rand()
        A = p(x_proposal, y_proposal) / p(x, y)
        if u < A: # accept with probability min(1, A)
            x, y = x_proposal, y_proposal
        samples[t+1] = np.array([x, y])

    return samples



if __name__ == "__main__":
    # set seed
    np.random.seed(3)

    # location and scale parameters
    mean = np.array([2, 2])
    cov = 0.5 * np.array([[1, 0.5], [0.5, 1]])

    # construct Markov chain
    def ppgauss(x, y):
        return pgauss(x, y, mean, cov)
    mc_samples = metropolis_hastings(ppgauss, nsamples=500)

    # sampling points
    nsamples = 5000
    data = np.random.multivariate_normal(mean, cov, nsamples)
    
    # construct contour data
    x = np.linspace(-2,5,500)
    y = np.linspace(-2,5,500)
    X,Y = np.meshgrid(x,y)

    pos = np.array([X.flatten(),Y.flatten()]).T
    rv = multivariate_normal(mean, cov)

    # figsize
    plt.figure(figsize=(20, 12))

    # plot samples
    ax = sns.scatterplot(x=data[:,0], y=data[:, 1],
                         alpha=0.4)

    # plotting contours
    ax.contour(x, y, rv.pdf(pos).reshape(500,500),
               levels=[0.03, 0.15, 0.3], 
               linewidths=2.5, colors=['black', 'grey', 'darkgray'])
    
    # ticks label size
    ax.tick_params(axis='both', which='major', labelsize=22)

    # coordinate system settings
    plt.axhline(0, color='black', linewidth=1.)
    plt.axvline(0, color='black', linewidth=1.)
    plt.ylim(-1.2, 4.5)
    plt.xlim(-1.2, 3.8)
    ax.set(frame_on=False)

    # save plot
    plt.savefig("../graphics/toy-mcmc.pdf", bbox_inches='tight', pad_inches=0, transparent=True)
    
    
    # add first 70 Markov chain samples
    plt.plot(mc_samples[:70,0],mc_samples[:70, 1], color='brown', marker='o', linewidth=2.25)
    
    # save plot
    plt.savefig("../graphics/toy-mcmc-with-samples.pdf", bbox_inches='tight', pad_inches=0, transparent=True)

    # add remaining Markov chain samples
    plt.plot(mc_samples[70:,0],mc_samples[70:, 1], color='brown', marker='o', linewidth=2.25)
    
    # save plot
    plt.savefig("../graphics/toy-mcmc-with-all-samples.pdf", bbox_inches='tight', pad_inches=0, transparent=True)

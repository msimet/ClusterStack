"""
model.py: contains definitions of the StackedModel classes for clusterstack. (c) 2018 M. Simet,
distributed under a BSD v3.0 license.
"""

import numbers
import inspect
import numpy as np

class StackedModel(object):
    r"""
    StackedModel: A class to hold information about clusters and compute stacked lens models
    (i.e., models that mimic the lens signal you'd get from a cross-correlation measurement
    method) and likelihoods for those models given a measured lensing signal.

    To use StackedModel, initialize the object with some information about the clusters
    (richness, redshift, lens weights, and a set of cosmological parameters) and a set of keyword
    arguments describing what you'd like to include in the model. We will start by describing the
    most basic model.  For richness :math:`\lambda`, redshift :math:`z`, and weights :math:`w`
    for a set of clusters indexed by :math:`i`, we start by computing a mass for each cluster
    :math:`M_i`, using a pivot richness :math:`\lambda_p` and a lognormal scatter
    :math:`\sigma_{\ln M|\lambda}`:

    .. math ::
        \log_{10} M_i = \log_{10} M_0 + \alpha \log_{10}\left(\frac{\lambda_i}{\lambda_p}\right)
            + \mathcal{N}(0, (\sigma_{\ln M|\lambda}+\alpha^2/\lambda_i)/\ln(10))

    (The final ln(10) is because the scatter width as given is in ln space not log10 space.)
    Then we compute a mass-concentration relation with some scatter as well, parameterized by
    :math:`\sigma_{{\rm log} c}`, which is in :math:`\log_{10}` space:

    .. math ::
        c_i = f(M_i)\times10^{\mathcal{N}(0, \sigma_{{\rm log} c})}

    From these, we compute a stacked and weighted model

    .. math ::
        \Delta\Sigma_{\rm stack}(R) =
            \frac{\sum\limits_i w_i \Delta\Sigma(R, M_i, c_i, z_i)}{\sum\limits_i w_i}

    For all cases where scatter is included, we do Monte Carlo realizations of the scatter (that is,
    simply draw Gaussian random numbers with the correct mean and variance).  Future versions may
    add an option to integrate over the distributions.

    Parameters
    ----------
    lens_richness : array_like
        a vector of some cluster quantity with a powerlaw association with mass,
        such as optical richness, with one entry for each cluster
    lens_redshift : array_like
        a vector of cluster redshifts, same order as lens_richness
    lens_weights : array_like
        a vector of lens weights, same order as lens_richness
    deltasigma_model_func : callable
        a callable function that will return delta sigma models
        given inputs; should have the same call signature as OffsetNFW
        functions
    richness_pivot : optional
        the pivot value of the observable for the mass-observable relation
    redshift_pivot : optional
        the pivot redshift for the redshift-dependent component of the
        mass-observable relation
    concentration_logscatter : optional
        the log10-space scatter of the mass-concentration relation
    concentration_mass_func
        a callable that takes mass and redshift and returns
        concentration; must be vectorized
    log10M_lim : tuple, optional
        the prior limits of the log10M parameter; for a tophat this is
        the low and high end, for the Gaussian it is the mean and variance
    log10M_lim_type : ['tophat', 'gaussian'], optional
        what type of prior to use for log10M
    alpha_lim : tuple, optional
        the prior limits of the alpha [M-richness powerlaw index] parameter
    alpha_lim_type : ['tophat', 'gaussian'], optional
    sigma_lnM_lim : tuple, optional
        the prior limits of the sigma_lnM parameter (log-space <M|observable> scatter)
    sigma_lnM_type : ['tophat', 'gaussian'], optional
    """
    # pylint: disable=too-many-instance-attributes
    # Some fragmentation is required to avoid duplication of code in inheritance.
    def __init__(self, lens_richness, lens_redshift, lens_weights,
                 deltasigma_model_func=None, 
                 richness_pivot=40, redshift_pivot=0.2,
                 concentration_logscatter=0.14, concentration_mass_func=None,
                 log10M_lim=(13, 15), log10M_lim_type='tophat',
                 alpha_lim=(0, 2), alpha_lim_type='tophat',
                 sigma_lnM_lim=(0.2, 0.3), sigma_lnM_lim_type='tophat'):
        # pylint: disable=too-many-arguments, too-many-locals
        try:
            self.arglist = str(inspect.signature(self.deltasigma))[1:-1].split(', ')[1:]
        except AttributeError:
            self.arglist = inspect.getargspec(self.deltasigma).args[2:]
        self.ln10 = np.log(10.)

        self.lens_richness = np.atleast_1d(lens_richness)
        self.lens_redshift = np.atleast_1d(lens_redshift)
        self.lens_weights = np.atleast_1d(lens_weights)[:, None] # add an extra dimension for later
        self.inverse_sum_weights = 1./np.sum(self.lens_weights)
        if not hasattr(deltasigma_model_func, '__call__'):
            raise TypeError("deltasigma_model_func must be a function")
        self.dsfunc = deltasigma_model_func
        self.richness_pivot = richness_pivot
        self.redshift_pivot = redshift_pivot
        self.cscatter = concentration_logscatter
        if not hasattr(concentration_mass_func, '__call__'):
            raise TypeError("concentration_mass_func must be a function")
        self.cmfunc = concentration_mass_func
        self.gaussians = {}
        self.tophats = {}
        self._update_priors('log10M', log10M_lim, log10M_lim_type)
        self._update_priors('alpha', alpha_lim, alpha_lim_type)
        self._update_priors('sigma_lnM', sigma_lnM_lim, sigma_lnM_lim_type)

    def _update_priors(self, key, lim, limtype):
        """ Update the one of the two prior dicts (self.gaussians or self.tophats) based on
        user input.

        The format of these dicts is that the key is the order within the arguments list (ie
        "theta" in emcee examples) and the values are the tophat limits (tophat) or mean and sigma
        for the Gaussian prior (gaussian)."""
        if limtype.lower() not in ['tophat', 'gaussian']:
            raise TypeError("{}_lim_type must be one of 'tophat' or 'gaussian'".format(key))
        if not hasattr(lim, '__len__') or len(lim) != 2:
            raise TypeError("{}_lim must be a 2-item tuple".format(key))
        # Want to avoid, say, string-comparison bugs
        if not (isinstance(lim[0], numbers.Number) and isinstance(lim[1], numbers.Number)):
            raise TypeError("{}_lim must be a 2-item tuple of numbers".format(key))
        if limtype.lower() == 'tophat':
            self.tophats[self.arglist.index(key)] = lim
        else:
            self.gaussians[self.arglist.index(key)] = lim

    def _scattermodel(self, scatter, alpha):
        """ Return the effective scatter for a given sigma_lnM and alpha. """
        # so we get ln m scatter but use log m
        return np.sqrt(scatter**2 + alpha**2/self.lens_richness)/self.ln10

    def _massmodel(self, log10M, alpha, sigma_lnM):
        """ Return a properly scattered mass given richness and the parameters log10M, alpha, and
            sigma_lnM. """
        log10M = np.atleast_1d(log10M)
        scatter = self._scattermodel(sigma_lnM, alpha)
        scatter_offset = np.array([np.random.normal(scale=s) for s in scatter])
        m = 10**(log10M+scatter_offset)
        masses = (m*(self.lens_richness/self.richness_pivot)**alpha)
        return masses

    def deltasigma(self, r, log10M, alpha, sigma_lnM):
        """ Produce a stacked delta sigma model.  A full mathematical description is given in the
        docstring for StackedModel.

        Parameters
        ----------
        r: array_like
            Sampling points for the radial bins
        log10M
            The base-10 log of the amplitude of the mass-observable powerlaw relation
        alpha
            The powerlaw index of the mass-observable powerlaw relation
        sigma_lnm
            The ln-space scatter of the mass given the observable
        """
        masses = self._massmodel(log10M, alpha, sigma_lnM)
        concentration = self.cmfunc(masses, self.lens_redshift)
        concentration *= 10**np.random.normal(scale=self.cscatter, size=concentration.shape)
        ds = self.dsfunc(r, masses, concentration, self.lens_redshift)
        weighted_ds = np.sum(self.lens_weights*ds, axis=0)*self.inverse_sum_weights
        return weighted_ds

    def lnprior(self, theta):
        """ The log prior of the parameter set theta, given the initialized prior ranges. """
        for index in self.tophats:
            if (theta[index] < self.tophats[index][0]) or (theta[index] > self.tophats[index][1]):
                return -np.inf
        lnprior = 0
        for index in self.gaussians:
            g = self.gaussians[index]
            lnprior -= (theta[index]-g[0])**2/(2*g[1]**2)
        return lnprior

    # This is basically boilerplate code from the emcee examples,
    # http://dfm.io/emcee/current/user/line/
    def lnlike(self, theta, x, y, yerr):
        """ Log likelihood. """
        return -np.sum((y-self.deltasigma(x, *theta))**2/(2*(yerr)**2))

    def lnprob(self, theta, x, y, yerr):
        """ Log probability (that is, log likelihood plus priors); useful for emcee runs. """
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        like = self.lnlike(theta, x, y, yerr)
        if np.any(np.isnan(like)):
            return -np.inf
        return lp + like

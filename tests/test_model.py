"""
Test various aspects of the creation and usage of StackedModel classes.  (c) 2018 M. Simet,
distributed under a BSD v3.0 license.
"""

# pylint: disable=missing-docstring
# pylint: disable=no-member
import numpy as np
try:
    from clusterstack import StackedModel
except ImportError:
    import sys
    sys.path.append('..')
    from clusterstack import StackedModel

richness = np.linspace(10, 100, num=66)
redshift = np.repeat(np.linspace(0.2, 0.3, num=22), 3)
weights = np.repeat(np.linspace(0.9, 1.1, num=33), 2)

def blankmodel(*args, **kwargs):
    # pylint: disable=unused-argument
    return np.zeros((len(args[1]), len(args[0])))
def dsmodel(rad, m, c, z):
    return rad*m[:, None]*c[:, None]*z[:, None]
def cmodel(m, z):
    # pylint: disable=unused-argument
    return m/0.5E13
r = np.linspace(0.1, 3.0, 10)


def check_limit_argument(key):
    """Check various permutations of prior limits & limit types"""
    np.testing.assert_raises(TypeError, StackedModel, richness, redshift, weights, blankmodel,
                             concentration_mass_func=blankmodel,
                             **{key: ('hello', 'goodbye')})
    StackedModel(richness, redshift, weights, blankmodel,
                 concentration_mass_func=blankmodel,
                 **{key: (-np.inf, np.inf)})
    np.testing.assert_raises(TypeError, StackedModel, richness, redshift, weights, blankmodel,
                             concentration_mass_func=blankmodel, **{key: (1, 2, 3)})
    np.testing.assert_raises(TypeError, StackedModel, richness, redshift, weights, blankmodel,
                             concentration_mass_func=blankmodel, **{key: 4})

def test_creation():
    """ Check the StackedModel object creation methods """
    # Check that objects are created properly with good arguments
    obj = StackedModel(richness, redshift, weights, blankmodel,
                       concentration_mass_func=blankmodel)
    np.testing.assert_equal(obj.lens_richness, richness)
    np.testing.assert_equal(obj.lens_redshift, redshift)
    np.testing.assert_equal(obj.lens_weights, weights[:, None])
    np.testing.assert_equal(blankmodel, obj.dsfunc)
    np.testing.assert_equal(blankmodel, obj.cmfunc)
    obj = StackedModel(richness, redshift, weights, blankmodel,
                       richness_pivot=80, redshift_pivot=0.25,
                       concentration_logscatter=0.12, concentration_mass_func=blankmodel,
                       log10M_lim=(12, 16), log10M_lim_type='tophat',
                       alpha_lim=(1.0, 0.5), alpha_lim_type='gaussian',
                       sigma_lnM_lim=(0.15, 0.35), sigma_lnM_lim_type='tophat')
    np.testing.assert_equal(obj.lens_richness, richness)
    np.testing.assert_equal(obj.lens_redshift, redshift)
    np.testing.assert_equal(obj.lens_weights, weights[:, None])
    np.testing.assert_equal(blankmodel, obj.dsfunc)
    np.testing.assert_equal(blankmodel, obj.cmfunc)
    np.testing.assert_equal(80, obj.richness_pivot)
    np.testing.assert_equal(0.25, obj.redshift_pivot)
    np.testing.assert_equal(0.12, obj.cscatter)
    np.testing.assert_equal(obj.tophats, {0: (12, 16), 2: (0.15, 0.35)})
    np.testing.assert_equal(obj.gaussians, {1: (1.0, 0.5)})

    # Now some failing options
    np.testing.assert_raises(TypeError, StackedModel, redshift, weights,
                             deltasigma_model_func=blankmodel,
                             concentration_mass_func=blankmodel)
    np.testing.assert_raises(TypeError, StackedModel, richness, weights,
                             deltasigma_model_func=blankmodel,
                             concentration_mass_func=blankmodel)
    np.testing.assert_raises(TypeError, StackedModel, richness, redshift,
                             deltasigma_model_func=blankmodel,
                             concentration_mass_func=blankmodel)
    np.testing.assert_raises(TypeError, StackedModel, richness, redshift, weights,
                             concentration_mass_func=blankmodel)
    np.testing.assert_raises(TypeError, StackedModel, richness, redshift,
                             deltasigma_model_func=blankmodel,
                             concentration_mass_func=blankmodel)
    np.testing.assert_raises(TypeError, StackedModel, richness, redshift, weights,
                             deltasigma_model_func=blankmodel)
    np.testing.assert_raises(TypeError, StackedModel, richness, redshift, weights, 3,
                             concentration_mass_func=blankmodel)
    np.testing.assert_raises(TypeError, StackedModel, richness, redshift, weights, blankmodel,
                             concentration_mass_func=3)
    np.testing.assert_raises(TypeError, StackedModel, richness, redshift, weights, blankmodel,
                             concentration_mass_func=blankmodel, log10M_lim_type='chisq')
    np.testing.assert_raises(TypeError, StackedModel, richness, redshift, weights, blankmodel,
                             concentration_mass_func=blankmodel, alpha_lim_type='chisq')
    np.testing.assert_raises(TypeError, StackedModel, richness, redshift, weights, blankmodel,
                             concentration_mass_func=blankmodel,
                             sigma_lnM_lim_type='chisq')

    check_limit_argument('log10M_lim')
    check_limit_argument('alpha_lim')
    check_limit_argument('sigma_lnM_lim')

def test_scattermodel():
    """ Check the mass scatter model """
    obj = StackedModel(richness, redshift, weights, blankmodel,
                       concentration_mass_func=blankmodel)
    np.testing.assert_array_almost_equal_nulp(obj._scattermodel(0.0, 1.0),
                                              1./np.sqrt(richness)/np.log(10))
    np.testing.assert_array_almost_equal_nulp(obj._scattermodel(np.log(10), 0),
                                              np.ones_like(richness))
    np.testing.assert_array_almost_equal_nulp(obj._scattermodel(1.5, 2.7),
                                              np.sqrt(1.5**2+2.7**2/richness)/np.log(10))

def test_massmodel():
    """ Check the mass model """
    obj = StackedModel(richness, redshift, weights, blankmodel,
                       concentration_mass_func=blankmodel, richness_pivot=1)
    # No scatter (analytic)
    masses = obj._massmodel(13, 0.0, 0.0)
    np.testing.assert_equal(masses, np.full(len(richness), 10**13))
    np.random.seed(37)
    masses = obj._massmodel(13, 0.0, 1.0)
    expected_masses = np.array([9.46992967e+12, 1.96267447e+13, 1.41431742e+13, 2.72437467e+12,
                                4.56542623e+13, 2.69076006e+13, 1.32006489e+13, 6.38528254e+12,
                                2.61683676e+13, 4.37106398e+12, 1.70686280e+13, 3.41571277e+13,
                                1.68134208e+13, 9.38610378e+12, 9.65804972e+12, 1.04661961e+13,
                                4.25470373e+13, 6.64261382e+13, 1.49635523e+13, 1.21218690e+13,
                                7.38087388e+12, 4.23258831e+12, 7.65407527e+12, 1.26293347e+12,
                                4.91579680e+12, 1.86181217e+13, 3.39222571e+13, 1.79702380e+12,
                                6.73107332e+12, 3.57432295e+13, 9.90378463e+12, 8.71062866e+12,
                                1.55918652e+13, 1.29112981e+12, 7.97668131e+12, 5.09344062e+13,
                                4.52449992e+12, 5.80679168e+13, 1.06234177e+13, 1.42990879e+13,
                                1.06883292e+13, 1.43695661e+13, 7.39994753e+12, 1.28377172e+12,
                                5.35382266e+13, 6.60694707e+12, 2.61298745e+13, 1.40909519e+13,
                                3.51525321e+13, 3.39391170e+12, 2.59363737e+12, 6.89405880e+12,
                                1.10745343e+13, 6.17664817e+12, 1.75880141e+12, 4.92314202e+13,
                                7.39868818e+12, 7.00371125e+13, 9.23861680e+12, 4.53988545e+13,
                                4.34181666e+12, 6.66356686e+12, 1.53042350e+13, 9.42671825e+12,
                                5.31045763e+12, 2.90796178e+13])

    np.testing.assert_allclose(masses, expected_masses)
    masses = obj._massmodel(13, 0.23, 1.2)
    expected_masses = np.array([1.317218e+13, 1.223372e+13, 5.026150e+12, 3.643972e+13,
                                2.197267e+13, 5.536394e+12, 6.265238e+12, 3.466024e+12,
                                6.196488e+13, 2.686705e+14, 2.568787e+12, 5.382080e+12,
                                3.089368e+14, 7.465468e+12, 5.801602e+13, 6.539443e+12,
                                8.805433e+12, 6.297092e+13, 4.545617e+13, 7.194419e+13,
                                4.099793e+13, 7.087641e+13, 2.057494e+13, 4.997545e+12,
                                2.642513e+14, 2.642771e+13, 8.975729e+12, 2.528691e+13,
                                1.444041e+13, 2.369753e+13, 1.470326e+13, 4.025653e+13,
                                9.752157e+13, 2.620065e+12, 7.050584e+13, 9.529972e+13,
                                1.811064e+13, 3.661838e+14, 1.310202e+13, 5.869205e+13,
                                1.413529e+13, 1.211712e+13, 3.225126e+13, 5.209478e+13,
                                1.964606e+14, 3.449645e+13, 1.323041e+14, 2.624082e+13,
                                2.554455e+13, 5.415681e+12, 1.088918e+13, 3.932316e+13,
                                2.384754e+14, 9.342287e+12, 1.177073e+14, 9.159576e+12,
                                1.365031e+14, 5.658493e+13, 2.506596e+13, 2.404336e+14,
                                7.202743e+12, 1.088756e+13, 1.514914e+14, 6.145998e+13,
                                2.928695e+13, 4.549905e+13])

    np.testing.assert_allclose(masses, expected_masses, rtol=1E-6)

def test_deltasigma():
    """ Check the delta sigma model """
    obj = StackedModel(richness, redshift, np.ones_like(weights), dsmodel,
                       concentration_mass_func=cmodel, richness_pivot=1, concentration_logscatter=0)

    # Simple case (avoids scatter)
    ds = obj.deltasigma(r, 13, 0, 0)
    np.testing.assert_allclose(ds, np.sum(r*1E13*2*redshift[:, None], axis=0)/len(richness))
    ds = obj.deltasigma(r, 13, 1., 0.25)
    np.random.seed(38)
    expected_ds = np.array([2.530700e+15, 1.068518e+16, 1.883965e+16, 2.699413e+16,
                            3.514861e+16, 4.330308e+16, 5.145756e+16, 5.961204e+16, 6.776651e+16,
                            7.592099e+16])
    np.testing.assert_allclose(ds, expected_ds, rtol=1E-6)

    # add cscatter
    obj = StackedModel(richness, redshift, np.ones_like(weights), dsmodel,
                       concentration_mass_func=cmodel, richness_pivot=1)
    ds = obj.deltasigma(r, 13, 1., 0.25)
    expected_ds = np.array([2.944124e+15, 1.243074e+16, 2.191737e+16, 3.140399e+16,
                            4.089061e+16, 5.037723e+16, 5.986385e+16, 6.935047e+16, 7.883709e+16,
                            8.832371e+16])
    np.testing.assert_allclose(ds, expected_ds, rtol=1E-6)

    # add weights
    obj = StackedModel(richness, redshift, weights, dsmodel,
                       concentration_mass_func=cmodel, richness_pivot=1)
    ds = obj.deltasigma(r, 13, 1., 0.25)
    expected_ds = np.array([3.446843e+15, 1.455334e+16, 2.565983e+16, 3.676633e+16,
                            4.787282e+16, 5.897932e+16, 7.008582e+16, 8.119231e+16, 9.229881e+16,
                            1.034053e+17])
    np.testing.assert_allclose(ds, expected_ds, rtol=1E-6)

def test_lnprior():
    """ Check the lnprior method """
    obj = StackedModel(richness, redshift, weights, blankmodel,
                       concentration_mass_func=blankmodel)
    np.testing.assert_equal(obj.lnprior((13, 1.0, 0.25)), 0)
    np.testing.assert_equal(obj.lnprior((12, 1.0, 0.25)), -np.inf)
    np.testing.assert_equal(obj.lnprior((16, 1.0, 0.25)), -np.inf)
    np.testing.assert_equal(obj.lnprior((13, -1.0, 0.25)), -np.inf)
    np.testing.assert_equal(obj.lnprior((13, 3.0, 0.25)), -np.inf)
    np.testing.assert_equal(obj.lnprior((13, 1.0, 0.15)), -np.inf)
    np.testing.assert_equal(obj.lnprior((13, 1.0, 0.35)), -np.inf)
    np.testing.assert_equal(obj.lnprior((12, -1.0, 0.25)), -np.inf)
    np.testing.assert_equal(obj.lnprior((16, 1.0, 0.35)), -np.inf)
    np.testing.assert_equal(obj.lnprior((13, 3.0, 0.15)), -np.inf)
    np.testing.assert_equal(obj.lnprior((12, -1.0, 0.15)), -np.inf)

    obj = StackedModel(richness, redshift, weights, blankmodel,
                       concentration_mass_func=blankmodel,
                       log10M_lim=(14., 1.), log10M_lim_type='gaussian',
                       alpha_lim=(1, 0.5), alpha_lim_type='gaussian',
                       sigma_lnM_lim=(0.25, 0.05), sigma_lnM_lim_type='gaussian')
    np.testing.assert_equal(obj.lnprior((13, 1.0, 0.25)), -0.5)
    np.testing.assert_equal(obj.lnprior((14, 2.0, 0.25)), -2)
    np.testing.assert_almost_equal(obj.lnprior((14, 1.0, 0.275)), -0.125)
    np.testing.assert_equal(obj.lnprior((13, 2.0, 0.25)), -2.5)
    np.testing.assert_almost_equal(obj.lnprior((13, 1.0, 0.275)), -0.625)
    np.testing.assert_equal(obj.lnprior((14, 2.0, 0.275)), -2.125)
    np.testing.assert_equal(obj.lnprior((13, 2.0, 0.275)), -2.625)

def test_lnlike():
    """ Check the lnlike method """
    obj = StackedModel(richness, redshift, np.ones_like(weights), dsmodel,
                       concentration_mass_func=cmodel, concentration_logscatter=0)
    y = np.sum(r*1E13*2*redshift[:, None], axis=0)/len(richness)
    yerr = 0.1
    # Note: these decimal vals are because we're adding a bunch of things 1E13 and then subtracting
    # them; errors of a few decimal places are still fine!
    np.testing.assert_almost_equal(obj.lnlike((13, 0, 0), r, y, yerr), 0, decimal=3)
    yerr = 0.1*np.ones_like(y)
    np.testing.assert_almost_equal(obj.lnlike((13, 0, 0), r, y, yerr), 0, decimal=3)
    np.testing.assert_almost_equal(obj.lnlike((13, 0, 0), r, 0, np.sqrt(0.5)), -np.sum(y**2),
                                   decimal=3)
    y -= 0.1
    np.testing.assert_almost_equal(obj.lnlike((13, 0, 0), r, y, yerr), -0.5*len(y), decimal=2)

def test_lnprob():
    """ Check the lnprob method """
    obj = StackedModel(richness, redshift, np.ones_like(weights), dsmodel,
                       concentration_mass_func=cmodel, concentration_logscatter=0,
                       sigma_lnM_lim=(0., 0.1), sigma_lnM_lim_type='gaussian')
    y = np.sum(r*1E13*2*redshift[:, None], axis=0)/len(richness)
    yerr = 0.1
    np.testing.assert_almost_equal(obj.lnprob((13, 0, 0), r, y, yerr), 0, decimal=3)
    np.testing.assert_almost_equal(obj.lnprob((13, 0, 0), r, 0, np.sqrt(0.5)), -np.sum(y**2),
                                   decimal=3)
    y -= 0.1
    np.testing.assert_almost_equal(obj.lnprob((13, 0, 0), r, y, yerr), -0.5*len(y), decimal=2)

    obj = StackedModel(richness, redshift, np.ones_like(weights), dsmodel,
                       concentration_mass_func=cmodel, concentration_logscatter=0,
                       log10M_lim=(14, 16), log10M_lim_type='tophat',
                       sigma_lnM_lim=(0., 0.1), sigma_lnM_lim_type='gaussian')
    y = np.sum(r*1E13*2*redshift[:, None], axis=0)/len(richness)
    yerr = 0.1
    np.testing.assert_equal(obj.lnprob((13, 0, 0), r, y, yerr), -np.inf)
    np.testing.assert_equal(obj.lnprob((13, 0, 0), r, 0, np.sqrt(0.5)), -np.inf)
    y -= 0.1
    np.testing.assert_equal(obj.lnprob((13, 0, 0), r, y, yerr), -np.inf)

    obj = StackedModel(richness, redshift, np.ones_like(weights), dsmodel,
                       concentration_mass_func=cmodel, concentration_logscatter=0,
                       sigma_lnM_lim=(0.1, 0.1), sigma_lnM_lim_type='gaussian')
    y = np.sum(r*1E13*2*redshift[:, None], axis=0)/len(richness)
    yerr = 0.1
    np.testing.assert_almost_equal(obj.lnprob((13, 0, 0), r, y, yerr), -0.5, decimal=3)
    np.testing.assert_equal(obj.lnprob((13, 0, 0), r, 0, np.sqrt(0.5)), -np.sum(y**2)-0.5)
    y -= 0.1
    np.testing.assert_almost_equal(obj.lnprob((13, 0, 0), r, y, yerr), -0.5*len(y)-0.5, decimal=2)

if __name__ == '__main__':
    test_creation()
    test_scattermodel()
    test_massmodel()
    test_deltasigma()
    test_lnprior()
    test_lnlike()
    test_lnprob()

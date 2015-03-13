#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A collection of functions for doing common processing tasks."""

import re
import numpy as np
from astropy import log
from astroML.density_estimation import scotts_bin_width, freedman_bin_width,\
    knuth_bin_width, bayesian_blocks
from pandas import DataFrame
from scipy.stats import describe as sp_describe

__author__ = 'Simon Mutch'
__email__ = 'smutch.astro@gmail.com'
__version__ = '0.1.0'


def pretty_print_dict(d, fmtlen=30):

    """Pretty print a dictionary, dealing with recursion.

    *Args*:
        d : dict
            the dictionary to print

        fmtlen : int
            maximum length of dictionary key for print formatting
    """

    fmtstr = "%%%ds :" % fmtlen
    fmtstr_title = "\n%%%ds\n%%%ds" % (fmtlen, fmtlen)

    for k, v in d.iteritems():
        if isinstance(v, dict):
            print fmtstr_title % (k.upper(), re.sub('\S', '-', k))
            pretty_print_dict(v)
        else:
            print fmtstr % k, v


def ndarray_to_dataframe(arr, drop_vectors=False):

    """Convert numpy ndarray to a pandas DataFrame, dealing with N(>1)
    dimensional datatypes.

    *Args*:
        arr : ndarray
            Numpy ndarray

    *Kwargs*:
        drop_vectors : bool
            only include single value datatypes in output DataFrame

    *Returns*:
        df : DataFrame
            Pandas DataFrame
    """

    # Get a list of all of the columns which are 1D
    names = []
    for k, v in arr.dtype.fields.iteritems():
        if len(v[0].shape) == 0:
            names.append(k)

    # Create a new dataframe with these columns
    df = DataFrame(arr[names])

    if not drop_vectors:
        # Loop through each N(>1)D property and append each dimension as
        # its own column in the dataframe
        for k, v in arr.dtype.fields.iteritems():
            if len(v[0].shape) != 0:
                for i in range(v[0].shape[0]):
                    df[k+'_%d' % i] = arr[k][:, i]

    return df


def mass_function(mass, volume, bins, range=None, poisson_uncert=False,
                  return_edges=False, **kwargs):

    """Generate a mass function.

    *Args*:
        mass : array
            an array of 'masses'

        volume : float
            volume of simulation cube/subset

        bins : int or list or str
            If bins is a string, then it must be one of:
                | 'blocks'   : use bayesian blocks for dynamic bin widths
                | 'knuth'    : use Knuth's rule to determine bins
                | 'scott'    : use Scott's rule to determine bins
                | 'freedman' : use the Freedman-diaconis rule to determine bins

    *Kwargs*:
        range : len=2 list or array
            range of data to be used for mass function

        poisson_uncert : bool
            return poisson uncertainties in output array (default: False)

        return_edges : bool
            return the bin_edges (default: False)

        **kwargs
            passed to np.histogram call

    *Returns*:
        array of [bin centers, mass function vals]

        If poisson_uncert=True then array has 3rd column with uncertainties.

        If return_edges=True then the bin edges are also returned.

    *Notes*:
        The code to generate the bin_widths is taken from astroML.hist

    """

    if "normed" in kwargs:
        kwargs["normed"] = False
        log.warn("Turned off normed kwarg in mass_function()")

    if (range is not None and (bins in ['blocks',
                                        'knuth', 'knuths',
                                        'scott', 'scotts',
                                        'freedman', 'freedmans'])):
        mass = mass[(mass >= range[0]) & (mass <= range[1])]

    if isinstance(bins, str):
        log.info("Calculating bin widths using `%s' method..." % bins)
        if bins in ['blocks']:
            bins = bayesian_blocks(mass)
        elif bins in ['knuth', 'knuths']:
            dm, bins = knuth_bin_width(mass, True)
        elif bins in ['scott', 'scotts']:
            dm, bins = scotts_bin_width(mass, True)
        elif bins in ['freedman', 'freedmans']:
            dm, bins = freedman_bin_width(mass, True)
        else:
            raise ValueError("unrecognized bin code: '%s'" % bins)
        log.info("...done")

    vals, edges = np.histogram(mass, bins, range, **kwargs)
    width = edges[1]-edges[0]
    radius = width/2.0
    centers = edges[:-1]+radius
    if poisson_uncert:
        uncert = np.sqrt(vals.astype(float))

    vals = vals.astype(float) / (volume * width)

    if not poisson_uncert:
        mf = np.dstack((centers, vals)).squeeze()
    else:
        uncert /= (volume * width)
        mf = np.dstack((centers, vals, uncert)).squeeze()

    if not return_edges:
        return mf
    else:
        return mf, edges


def edges_to_centers(edges, width=False):

    """Convert **evenly spaced** bin edges to centers.

    *Args*:
        edges : ndarray
            bin edges

    *Kwargs*:
        width : bool
            also return the bin width

    *Returns*:
        centers : ndarray
            bin centers (size = edges.size-1)

        bin_width : float
            only returned if width = True

    """

    bin_width = edges[1] - edges[0]
    radius = bin_width * 0.5
    centers = edges[:-1] + radius

    if width:
        return centers, bin_width
    else:
        return centers


def describe(arr, **kwargs):

    """Run scipy.stats.describe and produce legible output.

    *Args*:
        arr : ndarray
            Numpy ndarray

    *Kwargs*:
        passed to scipy.stats.describe

    *Returns*:
        output of scipy.stats.describe
    """

    stats = sp_describe(arr)

    print("{:15s} : {:g}".format("size", stats[0]))
    print("{:15s} : {:g}".format("min", stats[1][0]))
    print("{:15s} : {:g}".format("max", stats[1][1]))
    print("{:15s} : {:g}".format("mean", stats[2]))
    print("{:15s} : {:g}".format("unbiased var", stats[3]))
    print("{:15s} : {:g}".format("biased skew", stats[4]))
    print("{:15s} : {:g}".format("biased kurt", stats[5]))

    return stats


def smooth_grid(grid, side_length, radius, filt="tophat"):

    """Smooth a grid by convolution with a filter.

    *Args*:
        grid : ndarray
            The grid to be smoothed

        side_length : float
            The side length of the grid (assumes all side lengths are equal)

        radius : float
            The radius of the smoothing filter

    *Kwargs*:
        filt : string
            The name of the filter.  Currently only "tophat" (real space) is
            implemented.  More filters will be added over time.

    *Returns*:
        smoothed_grid : ndarray
            The smoothed grid
    """

    # The tuple of implemented filters
    IMPLEMENTED_FILTERS = ("tophat",)

    # Check to see if this filter is implemented
    if filt not in IMPLEMENTED_FILTERS:
        raise NotImplementedError("Filter not implemented.")

    side_length, radius = float(side_length), float(radius)

    # Do the forward fft
    grid = np.fft.fftn(grid)

    # Construct a grid of k*radius values
    k = 2.0 * np.pi * np.fft.fftfreq(grid.shape[0],
                                     1/float(grid.shape[0])) / side_length
    k = np.meshgrid(k, k, k, sparse=True, indexing='ij')
    kR = np.sqrt(k[0]**2 + k[1]**2 + k[2]**2)*radius

    # Evaluate the convolution
    if filt == "tophat":
        fgrid = grid * 3.0 * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)
    fgrid[kR==0] = grid[kR==0]

    # Inverse transform back to real space
    grid = np.fft.ifftn(fgrid).real

    # Make sure fgrid is marked available for garbage collection
    del(fgrid)

    return grid


def power_spectrum(grid, side_length, n_bins):

    """Calculate the dimensionless power spectrum of a grid (G):

    .. math::

       \Delta = \frac{k^3}{2\pi^2 V} <|\hat G|^2>

    *Args*:
        grid : ndarray
            The grid from which to construct the power spectrum

        side_length : float
            The side length of the grid (assumes all side lengths are equal)

        n_bins : int
            The number of k bins to use

    *Returns*:
        kmean : ndarray
            The mean wavenumber of each bin

        power : ndarray
            The dimensionless power (:math:`\Delta`)

        uncert : ndarray
            The standard deviation of the power within each k bin
    """


    volume = side_length**3

    # do the FFT (note the normalising 1.0/N_cells factor)
    ft_grid = np.fft.fftn(grid) / float(grid.size)

    # generate a grid of k magnitudes
    k1d = 2.0*np.pi * np.fft.fftfreq(grid.shape[0],
                                     1/float(grid.shape[0])) / side_length
    k = np.meshgrid(k1d, k1d, k1d, sparse=True)
    k = np.sqrt(k[0]**2 + k[1]**2 + k[2]**2)

    # bin up the k magnitudes
    k_edges = np.logspace(np.log10(np.sort(np.abs(k1d))[1]),
                          np.log10(k1d.max()), n_bins+1)
    k_bin = np.digitize(k.flat, k_edges) - 1
    np.clip(k_bin, 0, n_bins-1, k_bin)
    k_bin.shape = k.shape

    # loop through each k magnitude bin and calculate the mean power, k and uncert
    kmean = np.zeros(n_bins)
    power = np.zeros(n_bins)
    uncert = np.zeros(n_bins)

    for ii in xrange(n_bins):
        sel = k_bin == ii
        val = k[sel]**3 * np.abs(ft_grid[sel])**2 / (2.0 * np.pi**2) * volume
        power[ii] = val.mean()
        uncert[ii] = val.std()
        kmean[ii] = k[sel].mean()

    return kmean, power, uncert

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for reading Meraxes output files."""

from ..munge import ndarray_to_dataframe

import numpy as np
import h5py as h5
from astropy import log
import pandas as pd

__author__ = 'Simon Mutch'
__email__ = 'smutch.astro@gmail.com'
__version__ = '0.1.1'

__gal_props_h_conv = {
    "Pos": lambda x, h: x/h,
    "Mvir": lambda x, h: x/h,
    "FOFMvir": lambda x, h: x/h,
    "Rvir": lambda x, h: x/h,
    "HotGas": lambda x, h: x/h,
    "MetalsHotGas": lambda x, h: x/h,
    "ColdGas": lambda x, h: x/h,
    "MetalsColdGas": lambda x, h: x/h,
    "Mcool": lambda x, h: x/h,
    "StellarMass": lambda x, h: x/h,
    "NewStars": lambda x, h: x/h,
    "BlackHoleMass": lambda x, h: x/h,
    "DiskScaleLength": lambda x, h: x/h,
    "MetalsStellarMass": lambda x, h: x/h,
    "EjectedGas": lambda x, h: x/h,
    "MetalsEjectedGas": lambda x, h: x/h,
    "Rcool": lambda x, h: x/h,
    "MergTime": lambda x, h: x/h,
    "Mag": lambda x, h: x+5.0*np.log10(h),
    "MagDust": lambda x, h: x+5.0*np.log10(h),
}

def _check_pandas():
    try:
        pd
    except NameError:
        raise ImportError("The pandas package must be available if"
                          " pandas=True.")


def read_gals(fname, snapshot=None, props=None, quiet=False, sim_props=False,
              pandas=False, h=None, indices=None):

    """ Read in a Meraxes hdf5 output file.

    Reads in the default type of HDF5 file generated by the code.

    *Args*:
        fname (str): Full path to input hdf5 master file.

    *Kwargs*:
        snapshot (int): The snapshot to read in.
                        (default: last present snapshot - usually z=0)

        props (list): A list of galaxy properties requested.
                      (default: All properties)

        quiet (bool): Suppress output info and status messages.
                      (default: False)

        sim_props (bool): Output some simulation properties as well.
                          (default = False)

        pandas (bool): Ouput a pandas dataframe instead of a numpy array.
                       (default = False)

        h (float): Hubble constant (/100) to convert the galaxy properties to.
                   If `None` then no conversion is made.
                   (default = None)

        indices (list or array): Indices of galaxies to be read.  If `None`
                                 then read all galaxies.
                                 (default = None)

    *Returns*:
        Array with the requested galaxies and properties.

        If sim_props==True then output is a tuple of form
        (galaxies, sim_props)
    """

    def __apply_offsets(G, dest_sel, counter):
        # Deal with any indices that need offsets applied
        try:
            G[dest_sel]['CentralGal'] += counter
        except ValueError:
            pass


    if pandas:
        _check_pandas()

    # Open the file for reading
    fin = h5.File(fname, "r")

    # Set the snapshot correctly
    if snapshot is None:
        snapshot = -1
    if snapshot < 0:
        present_snaps = np.asarray(fin.keys())
        selection = np.array([(p.find('Snap') == 0) for p in present_snaps])
        present_snaps = [int(p[4:]) for p in present_snaps[selection]]
        snapshot = sorted(present_snaps)[snapshot]

    if not quiet:
        log.info("Reading snapshot %d" % snapshot)

    # Select the group for the requested snapshot.
    snap_group = fin['Snap%03d' % (snapshot)]

    # How many cores have been used for this run?
    n_cores = fin.attrs['NCores'][0]

    # Grab the total number of galaxies in this snapshot
    ngals = snap_group.attrs['NGalaxies'][0]

    if ngals == 0:
        raise IndexError("There are no galaxies in snapshot {:d}!"
                         .format(snapshot))

    # Reset ngals to be the number of requested galaxies if appropriate
    if indices is not None:
        indices = np.array(indices, 'i')
        indices.sort()
        ngals = indices.shape[0]

    # Set the galaxy data type
    if props is not None:
        gal_dtype = snap_group['Core0/Galaxies'].value[list(props)[:]][0].dtype
    else:
        gal_dtype = snap_group['Core0/Galaxies'].dtype

    # Create a dataset large enough to hold all of the requested galaxies
    G = np.empty(ngals, dtype=gal_dtype)
    if not quiet:
        log.info("Allocated %.1f MB" % (G.itemsize*ngals/1024./1024.))

    # Loop through each of the requested groups and read in the galaxies
    if ngals > 0:
        counter = 0
        total_read = 0
        for i_core in xrange(n_cores):
            galaxies = snap_group['Core%d/Galaxies' % i_core]
            core_ngals = galaxies.size

            if (indices is None) and (core_ngals > 0):
                dest_sel = np.s_[counter:core_ngals+counter]
                galaxies.read_direct(G, dest_sel=dest_sel)

                __apply_offsets(G, dest_sel, counter)
                counter += core_ngals

            else:
                read_ind = np.compress((indices >= total_read) &
                                       (indices < total_read+core_ngals),
                                       indices) - total_read

                if read_ind.shape[0] > 0:
                    dest_sel = np.s_[counter:read_ind.shape[0]+counter]
                    bool_sel = np.zeros(core_ngals, 'bool')
                    bool_sel[read_ind] = True
                    G[dest_sel] = galaxies[bool_sel]

                    __apply_offsets(G, dest_sel, total_read)
                    counter += read_ind.shape[0]

                total_read += core_ngals

            if counter >= ngals:
                break

    # Apply any unit conversions
    if h is not None:
        h = float(h)
        if not quiet:
            log.info("Converting units to h = %.3f" % h)
        for p in gal_dtype.names:
            try:
                G[p] = __gal_props_h_conv[p](G[p], h)
            except KeyError:
                pass

    # Print some checking statistics
    if not quiet:
        log.info('Read in %d galaxies.' % len(G))

    # If requested convert the numpy array into a pandas dataframe
    if pandas:
        G = ndarray_to_dataframe(G)

    # Set some run properties
    if sim_props:
        properties = read_input_params(fname, h=h, quiet=quiet)
        properties["Redshift"] = snap_group.attrs["Redshift"]

    fin.close()

    if sim_props:
        return G, properties
    else:
        return G


def read_input_params(fname, h=None, quiet=False):
    """ Read in the input parameters from a Meraxes hdf5 output file.

    *Args*:
        fname (str): Full path to input hdf5 master file.

        h (float): Hubble constant (/100) to convert the galaxy properties to.
                   If `None` then no conversion is made.
                   (default = None)

    *Returns*:
        A dict containing all run properties.
    """

    def arr_to_value(d):
        for k, v in d.iteritems():
            if v.size is 1:
                d[k] = v[0]

    def visitfunc(name, obj):
        if isinstance(obj, h5.Group):
            props_dict[name] = dict(obj.attrs.items())
            arr_to_value(props_dict[name])

    if not quiet:
        log.info("Reading input params...")

    # Open the file for reading
    fin = h5.File(fname, 'r')

    group = fin['InputParams']

    props_dict = dict(group.attrs.items())
    arr_to_value(props_dict)
    group.visititems(visitfunc)

    # Update some properties
    if h is not None:
        if not quiet:
            log.info("Converting units to h = %.3f" % h)
        props_dict['BoxSize'] = group.attrs['BoxSize'][0] / h
        props_dict['PartMass'] = group.attrs['PartMass'][0] / h

    props_dict['Volume'] = props_dict['BoxSize']**3.0 *\
        props_dict['VolumeFactor']

    fin.close()

    return props_dict


def read_git_info(fname):
    """Read the git diff and ref saved in the master file.

    *Args*:
        fname (str):  Full path to input hdf5 master file.

    *Returns*:
        ref (str):  git ref of the model
        diff (str): git diff of the model
    """

    with h5.File(fname, 'r') as fin:
        gitdiff = fin['gitdiff'].value
        gitref = fin['gitdiff'].attrs['gitref'].copy()

    return gitref, gitdiff


def read_snaplist(fname, h=None):

    """ Read in the list of available snapshots from the Meraxes hdf5 file.

    *Args*:
        fname (str): Full path to input hdf5 master file.

    *Kwargs:*
        h (float): Hubble constant (/100) to convert the galaxy properties to.
                   If `None` then no conversion is made.
                   (default = None)

    *Returns*:
        snaps:      array of snapshots

        redshifts:  array of redshifts

        lt_times:   array of light travel times (Gyr)
    """

    zlist = []
    snaplist = []
    lt_times = []

    with h5.File(fname, 'r') as fin:
        for snap in fin.keys():
            try:
                zlist.append(fin[snap].attrs['Redshift'][0])
                snaplist.append(int(snap[-3:]))
                lt_times.append(fin[snap].attrs['LTTime'][0])
            except KeyError:
                pass

    lt_times = np.array(lt_times, dtype=float)
    if h is not None:
        log.info("Converting units to h = %.3f" % h)
        lt_times /= h

    return np.array(snaplist, dtype=int), np.array(zlist, dtype=float),\
        lt_times


def check_for_redshift(fname, redshift, tol=0.1):
    """Check a Meraxes output file for the presence of a particular
    redshift.

    *Args*:
        fname (str):  Full path to input hdf5 master file
        redshift (float):  Redshift value

    *Kwargs*:
        tol (float):  +- tolerance on redshift value present.  An error will be
                      thrown of no redshift within this tollerance is found.

    *Returns*:
        snapshot (int):      Closest snapshot
        redshift (float):    Closest corresponding redshift
    """

    snaps, z, lt_times = read_snaplist(fname)
    zs = z-redshift

    w = np.argmin(np.abs(zs))

    if np.abs(zs[w]) > tol:
        raise KeyError("No redshifts within tolerance found.")

    return int(snaps[w]), z[w]


def grab_redshift(fname, snapshot):

    """ Quickly grab the redshift value of a single snapshot from a Meraxes
    HDF5 file.

    *Args*:
        fname (str):  Full path to input hdf5 master file

        snapshot (int):  Snapshot for which the redshift is to grabbed

    *Returns*:
        redshift (float):   Corresponding redshift value
    """

    with h5.File(fname, 'r') as fin:
        if snapshot < 0:
            present_snaps = np.asarray(fin.keys())
            selection = np.array([(p.find('Snap') == 0) for p in present_snaps])
            present_snaps = [int(p[4:]) for p in present_snaps[selection]]
            snapshot = sorted(present_snaps)[snapshot]
        redshift = fin["Snap{:03d}".format(snapshot)].attrs["Redshift"][0]

    return redshift


def grab_unsampled_snapshot(fname, snapshot):

    """ Quickly grab the unsampled snapshot value of a single snapshot from a
    Meraxes HDF5 file.

    *Args*:
        fname (str):  Full path to input hdf5 master file

        snapshot (int):  Snapshot for which the unsampled value is to be
                         grabbed

    *Returns*:
        redshift (float):   Corresponding unsampled snapshot value
    """

    with h5.File(fname, 'r') as fin:
        redshift = fin["Snap{:03d}".format(snapshot)].attrs["UnsampledSnapshot"][0]

    return redshift


def read_firstprogenitor_indices(fname, snapshot, pandas=False):

    """ Read the FirstProgenitor indices from the Meraxes HDF5 file.

    *Args*:
        fname (str):  Full path to input hdf5 master file

        snapshot (int):  Snapshot from which the progenitors dataset is to be
                         read from.

        pandas (bool): Return a pandas series instead of a numpy array.
                       (default = False)


    *Returns*:
        fp_ind (array or series): FirstProgenitor indices
    """

    if pandas:
        _check_pandas()

    with h5.File(fname, 'r') as fin:

        # number of cores used for this run
        n_cores = fin.attrs["NCores"][0]

        # group in the master file for this snapshot
        snap_group = fin["Snap{:03d}".format(snapshot)]

        # group for the previous snapshot
        prev_snap_group = fin["Snap{:03d}".format(snapshot-1)]

        # number of galaxies in this snapshot
        n_gals = snap_group.attrs["NGalaxies"][0]

        # malloc the fp_ind array and an array that will hold offsets for
        # each core
        fp_ind = np.zeros(n_gals, 'i4')
        prev_core_counter = np.zeros(n_cores, 'i4')

        # calculate the offsets for each core
        prev_core_counter[0] = 0
        for i_core in xrange(n_cores-1):
            prev_core_counter[i_core+1] = \
                prev_snap_group["Core{:d}/Galaxies".format(i_core)].size
        prev_core_counter = np.cumsum(prev_core_counter)

        # loop through and read in the FirstProgenitorIndices for each core. Be
        # sure to update the value to reflect that we are making one big array
        # from the output of all cores. Also be sure *not* to update fp indices
        # that = -1.  This has special meaning!
        counter = 0
        for i_core in xrange(n_cores):
            ds = snap_group["Core{:d}/FirstProgenitorIndices".format(i_core)]
            core_nvals = ds.size
            if core_nvals > 0:
                dest_sel = np.s_[counter:core_nvals+counter]
                ds.read_direct(fp_ind, dest_sel=dest_sel)
                counter += core_nvals
                fp_ind[dest_sel][fp_ind[dest_sel] > -1] += \
                    prev_core_counter[i_core]

    if pandas:
        fp_ind = pd.Series(fp_ind)

    return fp_ind


def read_nextprogenitor_indices(fname, snapshot, pandas=False):

    """ Read the NextProgenitor indices from the Meraxes HDF5 file.

    *Args*:
        fname (str):  Full path to input hdf5 master file

        snapshot (int):  Snapshot from which the progenitors dataset is to be
                         read from.

        pandas (bool): Return a pandas series instead of a numpy array.
                       (default = False)

    *Returns*:
        np_ind: NextProgenitor indices
    """

    if pandas:
        _check_pandas()

    with h5.File(fname, 'r') as fin:

        # number of cores used for this run
        n_cores = fin.attrs["NCores"][0]

        # group in the master file for this snapshot
        snap_group = fin["Snap{:03d}".format(snapshot)]

        # number of galaxies in this snapshot
        n_gals = snap_group.attrs["NGalaxies"][0]

        # malloc the np_ind array
        np_ind = np.zeros(n_gals, 'i4')

        # loop through and read in the NextProgenitorIndices for each core. Be
        # sure to update the value to reflect that we are making one big array
        # from the output of all cores. Also be sure *not* to update np indices
        # that = -1.  This has special meaning!
        counter = 0
        for i_core in xrange(n_cores):
            ds = snap_group["Core{:d}/NextProgenitorIndices".format(i_core)]
            core_nvals = ds.size
            if core_nvals > 0:
                dest_sel = np.s_[counter:core_nvals+counter]
                ds.read_direct(np_ind, dest_sel=dest_sel)
                np_ind[dest_sel][np_ind[dest_sel] > -1] += counter
                counter += core_nvals

    if pandas:
        np_ind = pd.Series(np_ind)

    return np_ind


def read_descendant_indices(fname, snapshot, pandas=False):

    """ Read the Descendant indices from the Meraxes HDF5 file.

    *Args*:
        fname (str):  Full path to input hdf5 master file

        snapshot (int):  Snapshot from which the descendant dataset is to be
                         read from.

        pandas (bool): Return a pandas series instead of a numpy array.
                       (default = False)

    *Returns*:
        desc_ind: NextProgenitor indices
    """

    if pandas:
        _check_pandas()

    with h5.File(fname, 'r') as fin:

        # number of cores used for this run
        n_cores = fin.attrs["NCores"][0]

        # group in the master file for this snapshot
        snap_group = fin["Snap{:03d}".format(snapshot)]

        # group for the next snapshot
        next_snap_group = fin["Snap{:03d}".format(snapshot+1)]

        # number of galaxies in this snapshot
        n_gals = snap_group.attrs["NGalaxies"][0]

        # malloc the desc_ind array and an array that will hold offsets for
        # each core
        desc_ind = np.zeros(n_gals, 'i4')
        prev_core_counter = np.zeros(n_cores, 'i4')

        # calculate the offsets for each core
        prev_core_counter[0] = 0
        for i_core in xrange(n_cores-1):
            prev_core_counter[i_core+1] = \
                next_snap_group["Core{:d}/Galaxies".format(i_core)].size
        prev_core_counter = np.cumsum(prev_core_counter)

        # loop through and read in the DescendantIndices for each core. Be sure
        # to update the value to reflect that we are making one big array from
        # the output of all cores. Also be sure *not* to update desc indices
        # that = -1.  This has special meaning!
        counter = 0
        for i_core in xrange(n_cores):
            ds = snap_group["Core{:d}/DescendantIndices".format(i_core)]
            core_nvals = ds.size
            if core_nvals > 0:
                dest_sel = np.s_[counter:core_nvals+counter]
                ds.read_direct(desc_ind, dest_sel=dest_sel)
                counter += core_nvals
                desc_ind[dest_sel][desc_ind[dest_sel] > -1] += \
                    prev_core_counter[i_core]

    if pandas:
        desc_ind = pd.Series(desc_ind)

    return desc_ind


def read_grid(fname, snapshot, name):

    """ Read a grid from the Meraxes HDF5 file.

    *Args*:
        fname (str):  Full path to input hdf5 master file

        snapshot (int):  Snapshot from which the grid is to be read from.

        name (str):  Name of the requested grid

    *Returns*:
        grid (array):  The requested grid
    """

    with h5.File(fname, 'r') as fin:
        HII_dim = fin["InputParams/tocf"].attrs["HII_dim"][0]
        ds_name = "Snap{:03d}/Grids/{:s}".format(snapshot, name)
        try:
            grid = fin[ds_name][:]
        except KeyError:
            log.error("No grid called %s found in file %s ." % (name, fname))

    grid.shape = [HII_dim, ]*3

    return grid


def list_grids(fname, snapshot):

    """ List the available grids from a Meraxes HDF5 output file.

    *Args*:
        fname (str):  Full path to input hdf5 master file

        snapshot (int):  Snapshot for which the grids are to be listed.

    *Returns*:
        grids (list):  A list of the available grids
    """

    with h5.File(fname, 'r') as fin:
        group_name = "Snap{:03d}/Grids".format(snapshot)
        try:
            grids = fin[group_name].keys()
        except KeyError:
            log.error("No grids found for snapshot %d in file %s ." %
                      (snapshot, fname))

    return grids

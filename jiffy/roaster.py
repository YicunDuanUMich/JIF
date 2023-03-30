#!/usr/bin/env python
# encoding: utf-8
#
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. Written by
# Michael D. Schneider schneider42@llnl.gov.
# LLNL-CODE-742321. All rights reserved.
#
# This file is part of JIF. For details, see https://github.com/mdschneider/JIF
#
# Please also read this link – Our Notice and GNU Lesser General Public License
# https://github.com/mdschneider/JIF/blob/master/LICENSE
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the terms and conditions of the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
'''
@file jiffy roaster.py

Draw posterior samples of image source model parameters
'''
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import galsim
import emcee

import jiffy
from . import priors, detections

from scipy.optimize import minimize
from galsim.errors import GalSimFFTSizeError, GalSimError

class Roaster(object):
    '''
    Likelihood model for footprint pixel data given a parametric source model

    Only single epoch images are allowed.
    '''
    def __init__(self, config='../config/jiffy.yaml'):
        if isinstance(config, str):
            import yaml
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        self.config = config

        prior_form = None
        prior_module = None
        detection_correction_form = None
        detection_correction_module = None
        for arg_name in self.config['model']:
            if arg_name[:6] == 'prior_':
                if arg_name[6:] == 'form':
                    prior_form = self.config['model'][arg_name]
                if arg_name[6:] == 'module':
                    prior_module = self.config['model'][arg_name]
            elif arg_name[:21] == 'detection_correction_':
                if arg_name[21:] == 'form':
                    detection_correction_form = self.config['model'][arg_name]
                if arg_name[21:] == 'module':
                    detection_correction_module = self.config['model'][arg_name]
        prior_kwargs = dict()
        detection_correction_kwargs = dict()
        if 'prior' in self.config:
            for arg_name in self.config['prior']:
                prior_kwargs[arg_name] = self.config['prior'][arg_name]
        if 'detection_correction' in self.config:
            for arg_name in self.config['detection_correction']:
                detection_correction_kwargs[arg_name] = self.config['detection_correction'][arg_name]
        
        self.prior = priors.initialize_prior(prior_form, prior_module, **prior_kwargs)
        self.detection_correction = detections.initialize_detection_correction(
            detection_correction_form, detection_correction_module, **detection_correction_kwargs)
        
        np.random.seed(self.config['init']['seed'])

        self.num_sources = self.config['model']['num_sources']

        actv_params = self.config['model']['model_params'].split(' ')
        model_kwargs = dict({'active_parameters': actv_params})
        self.n_params = len(actv_params)

        model_class_name = self.config['model']['model_class']
        if 'model_module' in self.config['model']:
            model_module = __import__(self.config['model']['model_module'])
        else:
            model_module = __import__('jiffy.galsim_galaxy')
        self.src_models = [getattr(model_module, model_class_name)(config, **model_kwargs)
                           for i in range(self.num_sources)]

        # Initialize objects describing the pixel data in a footprint
        self.ngrid_x = 64
        self.ngrid_y = 64
        self.noise_var = 3e-10
        self.scale = 0.2
        self.gain = 1.0
        self.data = None
        self.mask = None
        self.bkg = None
        self.lnnorm = self._set_like_lnnorm()
        
        # This is decided at the beginning of roasting. True by default until then.
        self.good_initial_params = True

    def get_params(self):
        '''
        Make a flat array of active model parameters for all sources

        For use in MCMC sampling
        '''
        return np.array([m.get_params() for m in self.src_models]).ravel()

    def set_params(self, params):
        '''
        Set the active parameters for all sources from a flattened array
        '''
        valid_params = True
        for isrc in range(self.num_sources):
            imin = isrc * self.n_params
            imax = (isrc + 1) * self.n_params
            p_set = params[imin:imax]
            valid_params *= self.src_models[isrc].set_params(p_set)
        return valid_params

    def set_param_by_name(self, paramname, value):
        '''
        Set a galaxy or PSF model parameter by name

        Can pass a single value that will be set for all source models, or a
        list of length num_sources with unique values for each source.
        '''
        if hasattr(value, '__len__'):
            if len(value) == self.num_sources:
                for isrc, v in enumerate(value):
                    if np.issubdtype(type(v), np.floating):
                        self.src_models[isrc].set_param_by_name(paramname, v)
                    else:
                        raise ValueError('If passing iterable, each entry must be a number')
            else:
                raise ValueError('If passing iterable, must have length num_sources')
        elif np.issubdtype(type(value), np.floating):
            for isrc in range(self.num_sources):
                self.src_models[isrc].set_param_by_name(paramname, value)
        else:
            raise ValueError('Unsupported type for input value')
        return None

    def make_data(self, noise=None, real_galaxy_catalog=None):
        '''
        Make fake data from the current stored galaxy model

        @param noise Specify custom noise model. Use GaussianNoise if not provided.
        @param mag Specify a magnitude or magnitudes for the image. Use default 
            fluxes from parameter config file if not provided.
        '''
        image = self._get_model_image(real_galaxy_catalog=real_galaxy_catalog)
        if noise is None:
            if np.issubdtype(type(self.noise_var), np.floating):
                noise = galsim.GaussianNoise(sigma=np.sqrt(self.noise_var))
            elif issubclass(type(self.noise_var), np.ndarray):
                noise = galsim.VariableGaussianNoise(rng=None, var_image=self.noise_var)
        image.addNoise(noise)
        self.data = image.array
        return image

    def draw(self):
        '''
        Draw simulated data from the likelihood function
        '''
        return self.make_data()

    def import_data(self, pix_dat_array, noise_var, mask=1, bkg=0, scale=0.2, gain=1.0):
        '''
        Import the pixel data and noise variance for a footprint
        '''
        self.ngrid_y, self.ngrid_x = pix_dat_array.shape
        self.data = pix_dat_array
        self.noise_var = noise_var
        self.mask = mask
        self.bkg = bkg
        self.scale = scale
        self.gain = gain
        self.lnnorm = self._set_like_lnnorm()

    def initialize_param_values(self, param_file_name):
        '''
        Initialize model parameter values from config file
        '''
        try:
            import configparser
        except:
            import ConfigParser as configparser

        config = configparser.RawConfigParser()
        config.read(param_file_name)

        params = config.items('parameters')
        for paramname, val in params:
            vals = str.split(val, ' ')
            if len(vals) > 1: ### Assume multiple sources
                fval = [float(v) for v in vals[0:self.num_sources]]
            else:
                fval = float(val)
            self.set_param_by_name(paramname, fval)
        return None
    
    # Warning: Currently only works with IsolatedFootprintPrior
    def map_initialize(self, args):
        # Initial parameter values for the optimizer
        # Use sum of footprint image pixel values for flux estimate
        # The minimum true inst flux in my data set is about 0.108
        flux0 = max(self.data.sum(), 0.1)
        # Find the expected hlr conditioned on the flux level
        prior_cov_hlrFlux = np.linalg.inv(self.prior.inv_cov_hlrFlux)
        hlr0 = np.exp(self.prior.mean_hlrFlux[0] +
                     (prior_cov_hlrFlux[1,0] / prior_cov_hlrFlux[1,1]) *
                     (np.log(flux0) - self.prior.mean_hlrFlux[1])) # pixels
        # The minimum true hlr in my data set is about 0.0275,
        # which is close to the prior mean (0.0245) for a flux of 0.108.
        hlr0 = max(hlr0 * 0.2, 0.02) # convert to arcsec
        # nu, hlr (arcsec), e1, e2_scale, flux, dx (arcsec), dy (arcsec)
        # e2_scale is defined as: e2 / sqrt(1 - e1**2)
        x0 = [0., hlr0, 0., 0., flux0, 0, 0]
        
        bnds = [# Excessively low nu coupled with high hlr can cause rendering problems
               (-0.75, 3.99), # nu
               # hlr needs to be able to go below any given value of flux
               (1e-5, 0.5), # hlr in arcsec
                # e1**2 + e2**2 should be strictly < 1
               (-0.99, 0.99), (-0.99, 0.99), # e1, e2_scale
               # Negative fluxes aren't physical. Small values may be explored for negative-flux images.
               (1e-4, None), # flux
               (None, None), (None, None) # dx, dy
        ]
        opts = {'ftol': 1e-8, 'eps': 1e-5}
        
        # Find the negative log-posterior for a given parameter tuple
        def loss(x):
            nu, hlr, e1, e2_scale, flux, dx, dy = tuple(x)
            e2 = e2_scale * np.sqrt(1 - e1**2)
            params = np.array([nu, hlr, e1, e2, flux, dx, dy])
            lnpost = self(params)
            return -lnpost
        
        # Try a MAP fit
        # Default to x0 if the fit fails
        fit_succeeded = False
        params_opt = x0
        try:
            res = minimize(loss, x0, method='L-BFGS-B',
                           bounds=bnds, options=opts)
            if not res.success:
                res = minimize(loss, x0, method='SLSQP',
                               bounds=bnds, options=opts)
            fit_succeeded = res.success
            if args.verbose and not fit_succeeded:
                print('Optimizer failed to find MAP.')
        except GalSimFFTSizeError:
            if args.verbose:
                print('GalSimFFTSizeError encountered during MAP fit.')
        except GalSimError:
            if args.verbose:
                print('GalSimError other than GalSimFFTSizeError encountered during MAP fit.')
        if not fit_succeeded:
            if args.verbose:
                print('MAP initialization failed. Using naive initialization.')
        else:
            params_opt = res.x
        
        # Unpack the fit results
        nu_opt, hlr_opt, e1_opt, e2_scale_opt, flux_opt, dx_opt, dy_opt = tuple(params_opt)
        e2_opt = e2_scale_opt * np.sqrt(1 - e1_opt**2)
        params_opt = np.array([nu_opt, hlr_opt, e1_opt, e2_opt, flux_opt, dx_opt, dy_opt])
        # Initialize the model with these parameters
        valid_params = self.set_params(params_opt)
        
        self.map_params = params_opt
        self.map_succeeded = fit_succeeded
        
        return None
    
    def initialize_from_image(self, args):
        image = galsim.ImageF(self.data)
        
        flux = image.array.sum()
        if flux < jiffy.galsim_galaxy.K_PARAM_BOUNDS['flux'][0]:
            if args.verbose:
                print('Flux initialization from image failed - Total image flux too small.')
        elif flux > jiffy.galsim_galaxy.K_PARAM_BOUNDS['flux'][1]:
            if args.verbose:
                print('Flux initialization from image failed - Total image flux too large.')
        else:
            self.set_param_by_name('flux', flux)
        
        try:
            moments = image.FindAdaptiveMom()
        except galsim.hsm.GalSimHSMError:
            if args.verbose:
                print('HSM initialization failed.')
            return None

        params = {'e1': moments.observed_shape.e1,
                'e2': moments.observed_shape.e2,
                # FindAdaptiveMom() returns sigma and centroid in units of pixels,
                # but the image model expects these to be in units of arc.
                'hlr': moments.moments_sigma * self.scale,
                'dx': (moments.moments_centroid.x - image.true_center.x) * self.scale,
                'dy': (moments.moments_centroid.y - image.true_center.y) * self.scale}
        for paramname, paramvalue in params.items():
            if paramvalue < jiffy.galsim_galaxy.K_PARAM_BOUNDS[paramname][0]:
                if args.verbose:
                    print('HSM estimate for', paramname, 'too small.')
                    print('Setting to lowest admissible value.')
                paramvalue = jiffy.galsim_galaxy.K_PARAM_BOUNDS[paramname][0]
            elif paramvalue > jiffy.galsim_galaxy.K_PARAM_BOUNDS[paramname][1]:
                if args.verbose:
                    print('HSM estimate for', paramname, 'too large.')
                    print('Setting to highest admissible value.')
                paramvalue = jiffy.galsim_galaxy.K_PARAM_BOUNDS[paramname][1]
            self.set_param_by_name(paramname, paramvalue)

        return None

    def _get_model_image(self, real_galaxy_catalog=None):
        # Set up a blank template image
        model_image = galsim.ImageF(self.ngrid_x, self.ngrid_y,
                                    scale=self.scale, init_value=0.)
        
        # Try to draw all the sources on the template image
        for isrc in range(self.num_sources):
            if model_image is None: # Can happen if previous source could not render
                # Give up on rendering, as this parameter combination's likelihood cannot be rigorously evaluated
                break
            else:
                model_image = self.src_models[isrc].get_image(image=model_image,
                                                              gain=self.gain,
                                                              real_galaxy_catalog=real_galaxy_catalog)
        
        return model_image

    def lnprior(self, params):
        '''
        Evaluate the log-prior of the model parameters
        '''
        try:
            res = self.prior(params)
        except:
            # Assign 0 probability to parameter combinations that produce an unhandled exception in prior evaluation
            return -np.inf
        
        return res

    def _set_like_lnnorm(self):
        # This is -inf at locations where noise_var == 0
        logden = -0.5 * np.log(2 * np.pi * self.noise_var)

        if issubclass(type(self.noise_var), np.ndarray) and self.noise_var.size > 1:
            # Using a per-pixel variance plane
            # Treat pixels with zero variance as masked
            valid_pixels = self.noise_var != 0
            if self.mask is not None:
                valid_pixels &= self.mask.astype(bool)
            if np.sum(valid_pixels) == 0:
                return np.nan
            lnnorm = np.sum(logden[valid_pixels])
        else:
            # Using a constant variance over the entire image
            # Treat zero variance as a mask for the entire image
            if self.noise_var == 0:
                return np.nan
            elif self.mask is None:
                npix = self.ngrid_x * self.ngrid_y
            elif np.issubdtype(type(self.mask), np.number):
                npix = self.ngrid_x * self.ngrid_y * mask
            else:
                npix = np.sum(self.mask)
            if npix == 0:
                return np.nan
            lnnorm = npix * logden

        return float(lnnorm)

    def lnlike(self, params):
        '''
        Evaluate the log-likelihood of the pixel data in a footprint
        '''
        res = -np.inf

        try:
            model = self._get_model_image()
        except:
            # Assign 0 probability to parameter combinations that produce an unhandled exception in image rendering
            return -np.inf
        
        if model is not None:
            # Compute log-likelihood assuming independent Gaussian-distributed noise in each pixel
            delta = model.array - self.data
            if issubclass(type(self.noise_var), np.ndarray) and self.noise_var.size > 1:
                # Using a per-pixel variance plane
                # Treat zero variance pixels as masked
                valid_pixels = self.noise_var != 0
                if self.mask is not None:
                    valid_pixels &= self.mask.astype(bool)
                if np.sum(valid_pixels) == 0:
                    return 0
                sum_chi_sq = np.sum(delta[valid_pixels]**2 / self.noise_var[valid_pixels])
            else:
                # Using a constant variance over the entire image
                # Treat zero variance as a mask for the entire image
                if self.noise_var == 0:
                    return 0
                if self.mask is None:
                    sum_chi_sq = np.sum(delta**2) / self.noise_var
                else:
                    if np.sum(self.mask) == 0:
                        return 0
                    sum_chi_sq = np.sum(delta[self.mask.astype(bool)]**2) / self.noise_var
            
            res = -0.5 * sum_chi_sq + self.lnnorm
        
        if self.detection_correction:
            # Scale up the likelihood to account for the fact that we're only
            # looking at data examples that pass a detection algorithm
            try:
                detection_correction = self.detection_correction(params)
            except:
                # Assign 0 probability to parameter combinations that produce an unhandled exception in detection correction evaluation
                return -np.inf
            res += detection_correction
        
        res = float(res)
        return res

    def __call__(self, params):
        # Assign 0 probability to invalid parameter combinations
        lnp = -np.inf

        valid_params = self.set_params(params)
        if valid_params:
            # Compute the log-likelihood and log-prior.
            # Don't bother with the log-evidence because this doesn't depend on specific parameter choices.
            lnp = self.lnlike(params)
            lnp += self.lnprior(params)

        return lnp


def init_roaster(args):
    '''
    Initialize Roaster object, load data, and setup model
    '''
    import yaml
    import footprints

    with open(args.config_file) as config_file:
        config = yaml.safe_load(config_file)

    rstr = Roaster(config)

    dat, noise_var, mask, bkg, scale, gain = None, None, None, None, None, None
    def _load_array(item):
        if isinstance(item, str):
            item = np.load(item)
        return item
    if 'footprint' in config:
        fp = config['footprint']
        dat = _load_array(fp['image']) if 'image' in fp else None
        noise_var = _load_array(fp['variance']) if 'variance' in fp else None
        mask = _load_array(fp['mask']) if 'mask' in fp else None
        scale = _load_array(fp['scale']) if 'scale' in fp else None
        gain = _load_array(fp['gain']) if 'gain' in fp else None
        bkg = _load_array(fp['background']) if 'background' in fp else None
    elif 'io' in config and 'infile' in config['io']:
        dat, noise_var, mask, bkg, scale, gain = footprints.load_image(config['io']['infile'],
            segment=args.footprint_number, filter_name=config['io']['filter'])
    if dat is not None:
        rstr.import_data(dat, noise_var, mask=mask, bkg=bkg, scale=scale, gain=gain)

    if 'init' in config and 'init_param_file' in config['init']:
        rstr.initialize_param_values(config['init']['init_param_file'])
    if args.map_initialize:
        self.map_params = None
        self.map_succeeded = False
        rstr.map_initialize(args)
    elif args.initialize_from_image:
        rstr.initialize_from_image(args)

    return rstr

def run_sampler(args, sampler, p0, nsamples, rstr):
    burned_in_state = p0
    nburn = rstr.config['sampling']['nburn']
    if nburn > 0:
        if args.verbose:
            print('Burning in')
        burned_in_state = sampler.run_mcmc(p0, nburn, progress=args.show_progress_bar)
        sampler.reset()
    if args.verbose:
        print('Sampling')
    final_state = sampler.run_mcmc(burned_in_state, nsamples, progress=args.show_progress_bar)
    pps = sampler.get_chain()
    lnps = sampler.get_log_prob()
    return pps, lnps

def do_sampling(args, rstr, return_samples=False, write_results=True, moves=None):
    '''
    Execute MCMC sampling for posterior model inference
    '''
    omega_interim = rstr.get_params()
    if not np.isfinite(rstr(omega_interim)):
        rstr.good_initial_params = False
        if args.verbose:
            print('Bad initial chain parameters.')

    nvars = len(omega_interim)
    nsamples = rstr.config['sampling']['nsamples']
    nwalkers = rstr.config['sampling']['nwalkers']

    p0 = emcee.utils.sample_ball(omega_interim, 
                                 np.ones_like(omega_interim) * 0.01, nwalkers)

    if args.unparallelize:
        sampler = emcee.EnsembleSampler(nwalkers, nvars, rstr, moves=moves)
        pps, lnps = run_sampler(args, sampler, p0, nsamples, rstr)
    else:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, nvars, rstr, moves=moves, pool=pool)
            pps, lnps = run_sampler(args, sampler, p0, nsamples, rstr)
    
    if args.cluster_walkers:
        pps, lnps = cluster_walkers(pps, lnps,
            thresh_multiplier=args.cluster_walkers_thresh)

    if write_results:
        write_to_h5(args, pps, lnps, rstr)
    if return_samples:
        return pps, lnps
    else:
        return None

def cluster_walkers(pps, lnps, thresh_multiplier=1):
    '''
    Down-select emcee walkers to those with the largest mean posteriors

    Follows the algorithm of Hou, Goodman, Hogg et al. (2012)
    '''
    # print("Clustering emcee walkers with threshold multiplier {:3.2f}".format(
    #       thresh_multiplier))
    pps = np.array(pps)
    lnps = np.array(lnps)
    ### lnps.shape == (Nsteps, Nwalkers) => lk.shape == (Nwalkers,)
    lk = -np.mean(np.array(lnps), axis=0)
    nwalkers = len(lk)
    ndx = np.argsort(lk)
    lks = lk[ndx]
    d = np.diff(lks)
    thresh = np.cumsum(d) / np.arange(1, nwalkers)
    selection = d > (thresh_multiplier * thresh)
    if np.any(selection):
        nkeep = np.argmax(selection)
    else:
        nkeep = nwalkers
    # print("pps, lnps:", pps.shape, lnps.shape)
    pps = pps[:, ndx[0:nkeep], :]
    lnps = lnps[:, ndx[0:nkeep]]
    # print("New pps, lnps:", pps.shape, lnps.shape)
    return pps, lnps

def write_to_h5(args, pps, lnps, rstr):
    '''
    Save an HDF5 file with posterior samples from Roaster
    '''
    import os
    import h5py
    outfile = rstr.config['io']['roaster_outfile'] + '_seg{:d}.h5'.format(args.footprint_number)
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    hfile = h5py.File(outfile, 'w')

    ### Store outputs in an HDF5 (sub-)group so we don't always
    ### need a separate HDF5 file for every segment.
    group_name = 'Samples/footprint{:d}'.format(args.footprint_number)
    grp = hfile.create_group(group_name)

    paramnames = rstr.config['model']['model_params'].split()
    if rstr.num_sources > 1:
        paramnames = [p + '_src{:d}'.format(isrc) for isrc in range(rstr.num_sources)
                      for p in paramnames]

    ## Write the MCMC samples and log probabilities
    if 'post' in grp:
        del grp['post']
    post = grp.create_dataset('post',
                              data=np.transpose(np.dstack(pps), [2, 0, 1]))
    # pnames = np.array(rstr.src_models[0][0].paramnames)
    post.attrs['paramnames'] = paramnames
    if 'logprobs' in grp:
        del grp['logprobs']
    _ = grp.create_dataset('logprobs', data=np.vstack(lnps))
    hfile.close()
    return None


def initialize_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Draw interim samples of source model parameters via MCMC.')

    parser.add_argument('--config_file', type=str,
                        default='../config/jiffy.yaml',
                        help='Name of a configuration file listing inputs.')

    parser.add_argument('--footprint_number', type=int, default=0,
                        help='The footprint number to load from input.')

    parser.add_argument('--unparallelize', action='store_true',
                        help='Disable parallelizing during sampling.' +
                        ' Usually need to do this if running multiple separate fits in parallel.')

    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose messaging.')
    parser.add_argument('--show_progress_bar', action='store_true',
                        help='Show progress bar.')
    
    parser.add_argument('--map_initialize', action='store_true',
                        help='Use MAP fit to set initial parameter values.' +
                        ' So far only implemented for isolated galaxy footprints.')
    parser.add_argument('--initialize_from_image', action='store_true',
                        help='Use image characteristics estimated with HSM to set initial parameter values.' +
                        ' So far only tested on centered, isolated galaxies.' +
                        ' Superseded by --map_initialize.')
    
    parser.add_argument('--cluster_walkers', action='store_true',
                        help='Throw away outlier walkers.')
    parser.add_argument('--cluster_walkers_thresh', type=float, default=4,
                        help='Threshold multiplier for throwing away walkers.')

    return parser


def main():
    parser = initialize_arg_parser()
    args = parser.parse_args()

    rstr = init_roaster(args)
    do_sampling(args, rstr)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

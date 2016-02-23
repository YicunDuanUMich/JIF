#!/usr/bin/env python
# encoding: utf-8
"""
galsim_galaxy.py

Wrapper for GalSim galaxy models to use in MCMC.
"""
import os
import math
import copy
import numpy as np
from operator import add
import warnings
###
import galsim
import galsim.wfirst
###
import segments
import psf_model as pm

### Some telescope model parameters
k_telescopes = {
    "LSST": {
        "effective_diameter": 6.4, # meters
        "pixel_scale": 0.2,        # arcseconds / pixel
        # Exposure time for defining the zero point reference
        "exptime_zeropoint": 30.,  # seconds
        "zeropoint": 'AB',
        # Referenc filter name for defining the magnitude model parameter
        "ref_filter_mag_param": 'r'
    },
    "WFIRST": {
        "effective_diameter": galsim.wfirst.diameter * (1. - galsim.wfirst.obscuration), # meters
        "pixel_scale": galsim.wfirst.pixel_scale,       # arcseconds / pixel
        # Exposure time for defining the zero point reference
        "exptime_zeropoint": galsim.wfirst.exptime, # seconds
        "zeropoint": 'AB',
        # Referenc filter name for defining the magnitude model parameter
        "ref_filter_mag_param": 'r'
    }
}


### These SEDs do not go to long enough wavelengths for WFIRST bands
# k_SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
### From the Brown et al. (2014) atlas:
k_SED_names = ['NGC_0695_spec', 'NGC_4125_spec', 'NGC_4552_spec', 'CGCG_049-057_spec']

k_lsst_filter_names = 'ugrizy'
### 'Central' passband wavelengths in nanometers
k_lsst_filter_central_wavelengths = {'u':360., 'g':500., 'r':620., 'i':750.,
                                'z':880., 'y':1000.}

k_wfirst_filter_names = ['Z087', 'Y106', 'J129', 'H158', 'F184', 'W149']
### 'Central' passband wavelengths in nanometers
k_wfirst_filter_central_wavelengths = {'r':620., 'Z087':867., 'Y106':1100.,
    'J129':1300., 'H158':994., 'F184':1880., 'W149':1410.}
### TESTING
# k_wfirst_filter_names = k_lsst_filter_names
# k_wfirst_filter_central_wavelengths = k_lsst_filter_central_wavelengths

### Minimum brightness a magnitude parameter can take
k_mag_param_minval = 99.


k_spergel_paramnames = ['nu', 'hlr', 'e', 'beta']

### Numpy composite object types for the model parameters for galaxy images under different
### modeling assumptions.
k_galparams_type_sersic = [('redshift', '<f8'), ('n', '<f8'), ('hlr', '<f8'),
                           ('e', '<f8'), ('beta', '<f8')]
k_galparams_type_sersic += [('mag_sed{:d}'.format(i+1), '<f8')
                            for i in xrange(len(k_SED_names))]
k_galparams_type_sersic += [('dx', '<f8'), ('dy', '<f8')]


k_galparams_type_spergel = [('redshift', '<f8')] + [(p, '<f8')
                            for p in k_spergel_paramnames]
k_galparams_type_spergel += [('mag_sed{:d}'.format(i+1), '<f8')
                             for i in xrange(len(k_SED_names))]
k_galparams_type_spergel += [('dx', '<f8'), ('dy', '<f8')]


k_galparams_type_bulgedisk = [('redshift', '<f8')]
k_galparams_type_bulgedisk += [('{}_bulge'.format(p), '<f8')
                               for p in k_spergel_paramnames]
k_galparams_type_bulgedisk += [('{}_disk'.format(p), '<f8')
                               for p in k_spergel_paramnames]
k_galparams_type_bulgedisk += [('mag_sed{:d}_bulge'.format(i+1), '<f8')
    for i in xrange(len(k_SED_names))]
k_galparams_type_bulgedisk += [('mag_sed{:d}_disk'.format(i+1), '<f8')
    for i in xrange(len(k_SED_names))]
k_galparams_type_bulgedisk += [('dx_bulge', '<f8'), ('dy_bulge', '<f8')]
k_galparams_type_bulgedisk += [('dx_disk', '<f8'), ('dy_disk', '<f8')]


k_galparams_types = {
    "Sersic": k_galparams_type_sersic,
    "Spergel": k_galparams_type_spergel,
    "BulgeDisk": k_galparams_type_bulgedisk
}


### The galaxy models are initialized with these values:
k_galparams_defaults = {
    "Sersic": [(1., 3.4, 1.0, 0.1, np.pi/4, 22., k_mag_param_minval,
        k_mag_param_minval, k_mag_param_minval, 0., 0.)],
    "Spergel": [(1.,        # redshift
                 0.3,       # nu
                 1.0,       # hlr
                 0.1,       # e
                 np.pi/4,   # beta
                 17.,      # mag_sed1
                 k_mag_param_minval,   # mag_sed2
                 k_mag_param_minval,   # mag_sed3
                 k_mag_param_minval,   # mag_sed4
                 0.,        # dx
                 0.)],      # dy
    "BulgeDisk": [(1.,
        0.5, 0.6, 0.05, 0.0,
        -0.6, 1.8, 0.3, np.pi/4,
        22., k_mag_param_minval, k_mag_param_minval, k_mag_param_minval,
        k_mag_param_minval, 22., k_mag_param_minval, k_mag_param_minval,
        0., 0., 0., 0.)]
}


def select_psf_paramnames(model_paramnames):
    """
    Given a list of galaxy and PSF model parameter names, select just the PSF
    model parameter names.

    Assumes PSF parameters contain the string 'psf'.
    """
    return [p for p in model_paramnames if 'psf' in p]


def select_galaxy_paramnames(model_paramnames):
    """
    Given a list of galaxy and PSF model parameter names, select just the galaxy
    model parameter names.

    Assumes PSF parameters contain the string 'psf', while galaxy parameter
    names do not.
    """
    return [p for p in model_paramnames if 'psf' not in p]


def load_filter_file_to_bandpass(table, wavelength_scale=1.0,
                                 effective_diameter_meters=6.4,
                                 exptime_sec=30.):
    """
    Create a Galsim.Bandpass object from a lookup table

    @param table Either (1) the name of a file for reading the lookup table
                 values for a bandpass, or (2) an instance of a
                 galsim.LookupTable
    @param wavelength_scale The multiplicative scaling of the wavelengths in the
                            input bandpass file to get units of nm (not used if
                            table argument is a LookupTable instance)
    @param effective_diameter_meters The effective diameter of the telescope
                                     (including obscuration) for the zeropoint
                                     calculation
    @param exptime_sec The exposure time for the zeropoint calculation
    """
    if isinstance(table, str):
        dat = np.loadtxt(table)
        table = galsim.LookupTable(x=dat[:,0]*wavelength_scale, f=dat[:,1])
    elif not isinstance(table, galsim.LookupTable):
        raise ValueError("table must be a file name or galsim.LookupTable")
    bp = galsim.Bandpass(table)
    bp = bp.thin(rel_err=1e-4)
    return bp.withZeropoint(zeropoint='AB',
        effective_diameter=100. * effective_diameter_meters,
        exptime=exptime_sec)


def wrap_ellipticity_phase(phase):
    """
    Map a phase in radians to [0, pi) to model ellipticity orientation.
    """
    return (phase % np.pi)


def lsst_noise(random_seed, gain=2.1, read_noise=3.6, sky_level=720.):
    """
    See GalSim/examples/lsst.yaml

    gain: e- / ADU
    read_noise: rms of read noise in electrons (if gain > 0)
    sky_level: ADU / pixel
    """
    rng = galsim.BaseDeviate(random_seed)
    return galsim.CCDNoise(rng,
                           gain=gain,
                           read_noise=read_noise,
                           sky_level=sky_level)


def wfirst_noise(random_seed):
    """
    Deprecated in favor of GalSim WFIRST module

    From http://wfirst-web.ipac.caltech.edu/wfDepc/visitor/temp1927222740/results.jsp
    """
    rng = galsim.BaseDeviate(random_seed)
    exposure_time_s = 150.
    pixel_scale_arcsec = 0.11
    read_noise_e_rms = 0.5 #5.
    sky_background = 3.6e-2 #3.60382E-01 # e-/pix/s
    gain = 2.1 # e- / ADU
    return galsim.CCDNoise(rng, gain=gain,
        read_noise=(read_noise_e_rms / gain) ** 2,
        sky_level=sky_background / pixel_scale_arcsec ** 2 * exposure_time_s)


def wfirst_sky_background(filter_name, bandpass):
    """
    Calculate the approximate sky background in e-/pixel using the GalSim
    WFIRST module
    """
    sky_level = galsim.wfirst.getSkyLevel(bandpass)
    sky_level *= (1.0 + galsim.wfirst.stray_light_fraction)
    ### Approximate sky level in e-/pix, ignoring variable pixel scale
    ### See GalSim demo13.py
    sky_level *= galsim.wfirst.pixel_scale**2
    sky_level += galsim.wfirst.thermal_backgrounds[filter_name]*galsim.wfirst.exptime
    return sky_level


class GalSimGalaxyModel(object):
    """
    Parametric galaxy model from GalSim for MCMC.

    Derived originally from GalSim examples/demo1.py

    @param telescope_name       Name of the telescope to model. Used to identify
                                filter curves. [Default: "LSST"]
    @param pixel_scale_arcsec   Pixel scale for image models [Default: 0.11]
    @param noise                GalSim noise model. [Default: None]
    @param galaxy_model         Name of the parametric galaxy model
                                [Default: "Spergel"]
    @param active_parameters    List of the parameter names for sampling
    @param wavelength_meters    Wavelength in meters to set the scale for the
                                optics PSF [Default: 620e-9]
    @param primary_diam_meters  Diameter of the telescope primary [Default: 2.4]
    @param filters              List of galsim.Bandpass instances. This argument, if not 'None',
                                takes precedence over 'filter_names' and defines which filters can
                                be used with this galaxy model instance.
    @param filter_names         List of filter names to be used if the 'filters' parameter is not
                                specified. If supplied, the names in this list must match those
                                in the ../input directory with tables of bandpasses.
                                If neither 'filters' or 'filter_names' ares supplied, a Warning
                                is raised and subsequent execution may produce unexpected results.
    @param filter_wavelength_scale Multiplicative scaling to apply to input filter wavelenghts
    @param atmosphere           Simulate an (infinite exposure) atmosphere PSF? [Default: False]
    @param psf_model            Specification for the PSF model. Can be:
                                    1. a GalSim InterpolatedImage instance
                                    2. a PSFModel instance
                                    3. a name of a parametric model
    """
    ### Define a reference filter with respect to which magnitude parameters are defined
    ref_filter = 'r'
    def __init__(self,
                 telescope_name="LSST",
                 pixel_scale_arcsec=0.11, ### arcseconds
                 noise=None,
                 galaxy_model="Spergel",
                 active_parameters=['hlr'], #, 'e', 'beta'],
                 primary_diam_meters=2.4,
                 filters=None,
                 filter_names=None,
                 filter_wavelength_scale=1.0,
                 atmosphere=False,
                 psf_model=None):
        self.telescope_name = telescope_name
        self.pixel_scale = pixel_scale_arcsec
        # if noise is None:
        #     noise = galsim.GaussianNoise(sigma=30.)
        self.noise = noise
        self.galaxy_model = galaxy_model
        self.active_parameters = active_parameters
        self.active_parameters_galaxy = select_galaxy_paramnames(active_parameters)
        self.active_parameters_psf = select_psf_paramnames(active_parameters)
        self.primary_diam_meters = primary_diam_meters
        self.filters = copy.deepcopy(filters)
        self.filter_names = filter_names
        self.atmosphere = atmosphere
        self.psf_model = psf_model

        self.achromatic_galaxy = False ### TODO: Finish implementation of achromatic_galaxy feature

        ### Set GalSim galaxy model parameters
        self.params = np.core.records.array(k_galparams_defaults[galaxy_model],
            dtype=k_galparams_types[galaxy_model])
        self.paramtypes = k_galparams_types[galaxy_model]
        # self.paramnames = [p[0] for p in k_galparams_types[galaxy_model]]
        self.paramnames = self.active_parameters
        # self.n_params = len(self.paramnames)
        self.n_params = len(self.active_parameters)

        ### Setup the PSF model
        if isinstance(self.psf_model, np.ndarray):
            self.psf_model_type = 'InterpolatedImage'
            self.psf_model = galsim.InterpolatedImage(self.psf_model)
        elif isinstance(self.psf_model, pm.PSFModel):
            self.psf_model_type = 'PSFModel class'
        else:
            self.psf_model_type = 'Parametric'

        ### Set GalSim SED model parameters
        self._load_sed_files()
        ### Load the filters that can be used to draw galaxy images
        if self.filters is None:
            if self.filter_names is not None:
                self._load_filter_files(filter_wavelength_scale)
            else:
                warnings.warn("No filters available in GalSimGalaxyModel: supply \
                              'filters' or 'filter_names' argument")
        else:
            self.filter_names = self.filters.keys()
        ### Add the reference filter for defining the magnitude parameters
        path, filename = os.path.split(__file__)
        datapath = os.path.abspath(os.path.join(path, "../input/"))
        ref_filename = os.path.join(datapath, '{}_{}.dat'.format('LSST',
            GalSimGalaxyModel.ref_filter))
        self.filters['ref'] = load_filter_file_to_bandpass(ref_filename)

        self.gsparams = galsim.GSParams(
            folding_threshold=1.e-1, # maximum fractional flux that may be folded around edge of FFT
            maxk_threshold=2.e-1,    # k-values less than this may be excluded off edge of FFT
            xvalue_accuracy=1.e-1,   # approximations in real space aim to be this accurate
            kvalue_accuracy=1.e-1,   # approximations in fourier space aim to be this accurate
            shoot_accuracy=1.e-1,    # approximations in photon shooting aim to be this accurate
            minimum_fft_size=16)     # minimum size of ffts

    def _load_sed_files(self):
        """
        Load SED templates from files.

        Copied from GalSim demo12.py
        """
        path, filename = os.path.split(__file__)
        datapath = os.path.abspath(os.path.join(path, "../input/"))
        self.SEDs = {}
        for SED_name in k_SED_names:
            SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
            self.SEDs[SED_name] = galsim.SED(SED_filename, wave_type='Ang')
        return None

    def _load_filter_files(self, wavelength_scale=1.0):
        """
        Load filters for drawing chromatic objects.

        Makes use of the module-level dictionary `k_telescopes` with values for
        setting the zeropoints. Specifically, the type of zeropoint ('AB'),
        the effective diameter of the telescope, and the exposure time.

        Adapted from GalSim demo12.py

        @param wavelength_scale     Multiplicative scaling of the wavelengths
                                    input from the filter files to get
                                    nanometers from whatever the input units are
        """
        if self.telescope_name == "WFIRST":
            ### Use the Galsim WFIRST module
            self.filters = galsim.wfirst.getBandpasses(AB_zeropoint=True)
        else:
            ### Use filter information in this module
            path, filename = os.path.split(__file__)
            datapath = os.path.abspath(os.path.join(path, "../input/"))
            self.filters = {}
            for filter_name in self.filter_names:
                filter_filename = os.path.join(datapath, '{}_{}.dat'.format(
                    self.telescope_name, filter_name))
                self.filters[filter_name] = load_filter_file_to_bandpass(
                    filter_filename, wavelength_scale,
                    k_telescopes[self.telescope_name]['effective_diameter'],
                    k_telescopes[self.telescope_name]['exptime_zeropoint']
                )
                # print("BP {} zeropoint: {}".format(filter_name,
                #     self.filters[filter_name].zeropoint))
        return None

    def set_param_by_name(self, paramname, value):
        """
        Set a single parameter value using the parameter name as a key.

        Can set 'active' or 'inactive' parameters. So, this routine gives a
        way to set fixed or fiducial values of model parameters that are not
        used in the MCMC sampling in Roaster.

        @param paramname    The name of the galaxy or PSF model parameter to set
        @param value        The value to assign to the model parameter
        """
        if 'psf' in paramname and self.psf_model_type == "PSFModel class":
            self.psf_model.params[paramname][0] = value
        else:
            self.params[paramname][0] = value
        return None

    def get_param_by_name(self, paramname):
        """
        Get a single parameter value using the parameter name as a key.

        Can access 'active' or 'inactive' parameters.
        """
        if 'psf' in paramname and self.psf_model_type == "PSFModel class":
            p = self.psf_model.params[paramname][0]
        else:
            p = self.params[paramname][0]
        return p

    def set_params(self, p):
        """
        Take a list of (active) parameters and set local variables.

        We assume p is a list or flat numpy array with values listed in the
        same order as the parameter names in self.active_parameters (which
        is supplied on instantiation of a `GalSimGalaxyModel` object).

        If the PSF model for this instance is a `PSFModel` object, then the
        active parameters of the PSFModel should be appended to the list input
        here.

        For use in emcee.

        @param p    A list or array of galaxy (and PSF) model parameter values
        """
        for ip, pname in enumerate(self.active_parameters_galaxy):
            self.params[pname][0] = p[ip]
            # if 'mag_sed' in pname:
            #     ### Transform flux variables with exp -- we sample in ln(Flux)
            #     self.params[pname][0] = p[ip]
            # else:
            #     self.params[pname][0] = p[ip]
        if self.psf_model_type == "PSFModel class":
            ### Assumes the PSF parameters are appended to the galaxy parameters
            self.psf_model.set_params(p[len(self.active_parameters_galaxy):])
        return None

    def get_params(self):
        """
        Return a list of active model parameter values.

        @returns a flat array of model parameter values in the order specified
                 in the `active_parameters` argument to the `GalSimGalaxyModel`
                 constructor
        """
        p = self.params[self.active_parameters_galaxy].view('<f8').copy()
        if self.psf_model_type == "PSFModel class":
            psf_active_params = self.psf_model.get_params()
            if len(psf_active_params) > 0:
                p = np.append(p, self.psf_model.get_params())
        ### Transform fluxes to ln(Flux) for MCMC sampling
        for ip, pname in enumerate(self.active_parameters):
            if 'beta' in pname:
                p[ip] = wrap_ellipticity_phase(p[ip])
            # if 'flux_sed' in pname:
            #     p[ip] = np.log(p[ip])
        return p

    def validate_params(self):
        """
        Check that all model parameters take values inside allowed ranges.

        @returns a boolean indicating the validity of the current model
                 parameters
        """
        valid_params = True
        ### ===================================================================
        ### Parameters common to 'Sersic' and 'Spergel' parameterizations
        if self.galaxy_model == "Sersic" or self.galaxy_model == "Spergel":
            ### Redshift must be positive and less than a large value
            if self.params[0].redshift < 0.0 or self.params[0].redshift > 6.0:
                valid_params *= False
            ### Ellipticity must be on [0, 1]
            if self.params[0].e < 0. or self.params[0].e > 0.9:
                valid_params *= False
            ### Half-light radius must be positive and less than a large value
            ### (Large value here assumed in arcseconds)
            if self.params[0].hlr < 0.0 or self.params[0].hlr > 10.:
                valid_params *= False
            ### Position angle (in radians) must be on [0, pi]
            if self.params[0].beta < 0.0 or self.params[0].beta > np.pi:
                valid_params *= False
            # ### Flux must be strictly positive
            # for i in xrange(len(k_SED_names)):
            #     if self.params[0]['flux_sed{:d}'.format(i+1)] <= 0.:
            #         valid_params *= False
            ### Put a hard bound on the position parameters to avoid absurd
            ### translations of the galaxy
            if self.params[0].dx < -10. or self.params[0].dx > 10.:
                valid_params *= False
            if self.params[0].dy < -10. or self.params[0].dy > 10.:
                valid_params *= False
        ### ===================================================================
        if self.galaxy_model == "Spergel":
            if self.params[0].nu < -0.8 or self.params[0].nu > 0.6:
                valid_params *= False
        ### ===================================================================
        elif self.galaxy_model == "BulgeDisk":
            if (self.params[0].e_bulge < 0. or self.params[0].e_bulge > 1. or
                self.params[0].e_disk < 0. or self.params[0].e_disk > 1.):
                valid_params *= False
            if (self.params[0].nu_bulge < -0.6 or self.params[0].nu_bulge > 055 or
                self.params[0].nu_disk < -0.6 or self.params[0].nu_disk > 0.55):
                valid_params *= False
            # for i in xrange(len(k_SED_names)):
            #     if self.params[0]['flux_sed{:d}_bulge'.format(i+1)] <= 0.:
            #         valid_params *= False
            #     if self.params[0]['flux_sed{:d}_disk'.format(i+1)] <= 0.:
            #         valid_params *= False
        if self.psf_model_type == "PSFModel class":
            valid_params *= self.psf_model.validate_params()
        return valid_params

    def get_psf(self, filter_name='r'):
        """
        Get the PSF as a `GSObject` for use in GalSim image rendering or
        convolutions

        The type of PSF model is determined by the `psf_model` argument to the
        class constructor. The PSF object returned here could be:
            1. a GalSim `InterpolatedImage`
            2. a JIF `PSFModel`
            3. a GalSim model PSF composed of optics and, optionally,
               atmosphere components

        @returns the PSF model instance
        """
        if self.psf_model_type == 'InterpolatedImage':
            psf = self.psf_model
        elif self.psf_model_type == 'PSFModel class':
            psf = self.psf_model.get_psf()
        else:
            lam_over_diam = self.filters[filter_name].effective_wavelength*1.e-9 / self.primary_diam_meters
            lam_over_diam *= 206264.8 # arcsec
            optics = galsim.Airy(lam_over_diam, obscuration=0.548, flux=1.,
                gsparams=self.gsparams)
            if self.atmosphere:
                atmos = galsim.Kolmogorov(fwhm=0.6, gsparams=self.gsparams)
                psf = galsim.Convolve([atmos, optics])
            else:
                psf = optics
        return psf

    def get_SED(self, gal_comp='', flux_ref_wavelength=620.):
        """
        Get the GalSim SED object given the SED parameters and redshift.

        This routine passes galsim_galaxy magnitude parameters to the GalSim
        SED.withMagnitude() method.

        [Deprecated 2016-02-17]
        This routine passes galsim_galaxy flux parameters to the GalSim SED.withFluxDensity()
        method. The flux parameters therefore have units of photons/nm at a reference wavelength
        (here defined to be 620 nm) as required by GalSim.

        @param gal_comp             Name of the galaxy component (bulge,disk) to
                                    select. Can be the empty string to get the
                                    composite galaxy model SED.
        """
        if len(gal_comp) > 0:
            gal_comp = '_' + gal_comp
        bp =self.filters['ref']
        SEDs = [self.SEDs[SED_name].withMagnitude(
            target_magnitude=self.params[0]['mag_sed{:d}{}'.format(i+1, gal_comp)],
            bandpass=bp).atRedshift(self.params[0].redshift)
                for i, SED_name in enumerate(self.SEDs)]
        # SEDs = [self.SEDs[SED_name].withFluxDensity(
        #     target_flux_density=self.params[0]['flux_sed{:d}{}'.format(i+1, gal_comp)],
        #     wavelength=flux_ref_wavelength).atRedshift(self.params[0].redshift)
        #         for i, SED_name in enumerate(self.SEDs)]
        return reduce(add, SEDs)

    def get_flux(self, filter_name='r'):
        """
        Get the flux of the galaxy model in the named bandpass

        @param filter_name  Name of the bandpass for the desired magnitude

        @returns the flux in the requested bandpass (in photon counts)
        """
        if self.achromatic_galaxy:
            raise NotImplementedError()
        else:
            SED = self.get_SED()
            flux = SED.calculateFlux(self.filters[filter_name])
        return flux

    def get_magnitude(self, filter_name='r'):
        """
        Get the magnitude of the galaxy model in the named bandpass

        @param filter_name  Name of the bandpass for the desired magnitude

        @returns the magnitude in the requested bandpass
        """
        if self.achromatic_galaxy:
            raise NotImplementedError()
        else:
            SED = self.get_SED()
            mag = SED.calculateMagnitude(self.filters[filter_name])
        return mag

    def get_image(self, out_image=None, add_noise=False,
                  filter_name='r', gain=2.1, snr=None):
        if self.galaxy_model == "Gaussian":
            # gal = galsim.Gaussian(flux=self.params.gal_flux, sigma=self.params.gal_sigma)
            # gal_shape = galsim.Shear(g=self.params.e, beta=self.params.beta*galsim.radians)
            # gal = gal.shear(gal_shape)
            raise AttributeError("Unimplemented galaxy model")

        elif self.galaxy_model == "Spergel":
            mono_gal = galsim.Spergel(nu=self.params[0].nu,
                half_light_radius=self.params[0].hlr,
                # flux=self.params[0].gal_flux,
                flux=1.0,
                gsparams=self.gsparams)
            if self.achromatic_galaxy:
                gal = mono_gal
            else:
                SED = self.get_SED()
                gal = galsim.Chromatic(mono_gal, SED)
            gal_shape = galsim.Shear(g=self.params[0].e,
                beta=self.params[0].beta*galsim.radians)
            gal = gal.shear(gal_shape)
            gal = gal.shift(self.params[0].dx, self.params[0].dy)

        elif self.galaxy_model == "Sersic":
            mono_gal = galsim.Sersic(n=self.params[0].n,
                half_light_radius=self.params[0].hlr,
                # flux=self.params[0].gal_flux,
                flux=1.0,
                gsparams=self.gsparams)
            if self.achromatic_galaxy:
                gal = mono_gal
            else:
                SED = self.get_SED()
                gal = galsim.Chromatic(mono_gal, SED)
            SED = self.get_SED()
            gal = galsim.Chromatic(mono_gal, SED)
            gal_shape = galsim.Shear(g=self.params[0].e,
                beta=self.params[0].beta*galsim.radians)
            gal = gal.shear(gal_shape)
            gal = gal.shift(self.params[0].dx, self.params[0].dy)

        elif self.galaxy_model == "BulgeDisk":
            mono_bulge = galsim.Spergel(nu=self.params[0].nu_bulge,
                half_light_radius=self.params[0].hlr_bulge,
                flux=1.0,
                gsparams=self.gsparams)
            SED_bulge = self.get_SED(gal_comp='bulge')
            bulge = galsim.Chromatic(mono_bulge, SED_bulge)
            bulge_shape = galsim.Shear(g=self.params[0].e_bulge,
                beta=self.params[0].beta_bulge*galsim.radians)
            bulge = bulge.shear(bulge_shape)
            bulge = bulge.shift(self.params[0].dx_bulge, self.params[0].dy_bulge)

            mono_disk = galsim.Spergel(nu=self.params[0].nu_disk,
                half_light_radius=self.params[0].hlr_disk,
                flux=1.0,
                gsparams=self.gsparams)
            SED_disk = self.get_SED(gal_comp='disk')
            disk = galsim.Chromatic(mono_disk, SED_disk)
            disk_shape = galsim.Shear(g=self.params[0].e_disk,
                beta=self.params[0].beta_disk*galsim.radians)
            disk = disk.shear(disk_shape)
            disk = disk.shift(self.params[0].dx_disk, self.params[0].dy_disk)

            # gal = self.params[0].bulge_frac * bulge + (1 - self.params[0].bulge_frac) * disk
            gal = bulge + disk
            gal = gal.shift(dx, dy)

        else:
            raise AttributeError("Unimplemented galaxy model")
        final = galsim.Convolve([gal, self.get_psf(filter_name)])
        # wcs = galsim.PixelScale(self.pixel_scale)'

        try:
            image = final.drawImage(bandpass=self.filters[filter_name],
                image=out_image, scale=self.pixel_scale, gain=gain,
                add_to_image=False,
                method='fft')
            if add_noise:
                if self.telescope_name == "WFIRST":
                    sky_level = wfirst_sky_background(filter_name, self.filters[filter_name])
                    image += sky_level
                    galsim.wfirst.allDetectorEffects(image)
                    image -= (sky_level + galsim.wfirst.dark_current*galsim.wfirst.exptime) / galsim.wfirst.gain
                else:
                    if self.noise is not None:
                        if snr is None:
                            image.addNoise(self.noise)
                        else:
                            image.addNoiseSNR(self.noise, snr=snr)
                    else:
                        raise AttributeError("A GalSim noise model must be \
                                              specified to add noise to an image.")
        except RuntimeError:
            print "Trying to make an image that's too big."
            print self.get_params()
            image = None
        return image

    def get_psf_image(self, filter_name='r', ngrid=None):
        psf = self.get_psf(filter_name)
        if self.psf_model_type == 'InterpolatedImage':
            return psf
        elif self.psf_model_type == 'PSFModel class':
            return self.psf_model.get_psf_image(ngrid=ngrid,
                pixel_scale_arcsec=self.pixel_scale)
        else:
            if ngrid is None:
                ngrid = 16
            image_epsf = psf.drawImage(scale=self.pixel_scale, nx=ngrid, ny=ngrid)
            return image_epsf

    def get_segment(self):
        pass

    def save_image(self, file_name, out_image=None, filter_name='r'):
        image = self.get_image(filter_name=filter_name, out_image=out_image)
        image.write(file_name)
        return None

    def save_psf(self, file_name, ngrid=None, filter_name='r'):
        image_epsf = self.get_psf_image(filter_name, ngrid)
        image_epsf.write(file_name)
        return None

    def plot_image(self, file_name, ngrid=None, filter_name='r', title=None):
        import matplotlib.pyplot as plt
        if ngrid is not None:
            out_image = galsim.Image(ngrid, ngrid)
        else:
            out_image = None

        im = self.get_image(out_image, add_noise=True, filter_name=filter_name)
        print "Image rms: ", np.sqrt(np.var(im.array.ravel()))
        ###
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(im.array,
            cmap=plt.get_cmap('pink'), origin='lower',
            interpolation='none',
            extent=[0, ngrid*self.pixel_scale, 0, ngrid*self.pixel_scale])
        ax.set_xlabel(r"Detector $x$-axis (arcsec.)")
        ax.set_ylabel(r"Detector $y$-axis (arcsec.)")
        if title is not None:
            ax.set_title(title)
        cbar = fig.colorbar(im)
        cbar.set_label(r"photons / pixel")
        fig.savefig(file_name)
        return None

    def plot_psf(self, file_name, ngrid=None, title=None, filter_name='r'):
        import matplotlib.pyplot as plt
        psf = self.get_psf(filter_name)
        if ngrid is None:
            ngrid = 16
        image_epsf = psf.drawImage(image=None,
            scale=self.pixel_scale, nx=ngrid, ny=ngrid)
        ###
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(1,1,1)
        ngx, ngy = image_epsf.array.shape
        im = ax.imshow(image_epsf.array,
            cmap=plt.get_cmap('pink'), origin='lower',
            interpolation='none',
            extent=[0, ngx*self.pixel_scale, 0, ngy*self.pixel_scale])
        ax.set_xlabel(r"Detector $x$-axis (arcsec.)")
        ax.set_ylabel(r"Detector $y$-axis (arcsec.)")
        if title is not None:
            ax.set_title(title)
        cbar = fig.colorbar(im)
        cbar.set_label(r"normalized photons / pixel")
        fig.savefig(file_name)
        return None

    def get_moments(self, add_noise=True):
        results = self.get_image(add_noise=add_noise).FindAdaptiveMom()
        print 'HSM reports that the image has observed shape and size:'
        print '    e1 = %.3f, e2 = %.3f, sigma = %.3f (pixels)' % (results.observed_shape.e1,
                    results.observed_shape.e2, results.moments_sigma)


def save_bandpasses_to_segment(seg, gg, filter_names, telescope_name="LSST", scale=1):
    """
    Read filter files and copy to a 'segment' HDF5 file
    """
    path, filename = os.path.split(__file__)
    filter_names = list(filter_names)
    waves_nm_list = []
    throughputs_list = []
    effective_wavelengths = []
    for i, f in enumerate(filter_names):
        bp = np.loadtxt(os.path.join(path, "../input/{}_{}.dat".format(
            telescope_name, f)))
        waves_nm_list.append(bp[:,0]*scale)
        throughputs_list.append(bp[:,1])
        effective_wavelengths.append(gg.filters[f].effective_wavelength)
    # print "effective wavelengths (nm):", effective_wavelengths
    seg.save_bandpasses(filter_names,
        waves_nm_list, throughputs_list,
        effective_wavelengths=effective_wavelengths,
        telescope=telescope_name.lower())
    return None


def make_test_images(filter_name_ground='r', filter_name_space='F184',
                     file_lab='', galaxy_model="Spergel"):
    """
    Use the GalSimGalaxyModel class to make test images of a galaxy for LSST and WFIRST.
    """
    import os
    import h5py

    ngrid_lsst = 70

    print("Making test images for LSST and WFIRST")

    # LSST
    print("\n----- LSST -----")
    lsst = GalSimGalaxyModel(
        telescope_name="LSST",
        pixel_scale_arcsec=k_telescopes['LSST']['pixel_scale'],
        noise=lsst_noise(82357),
        galaxy_model=galaxy_model,
        primary_diam_meters=8.4,
        filter_names=k_lsst_filter_names,
        filter_wavelength_scale=1.0,
        atmosphere=True)

    # Save the image
    lsst.save_image("../TestData/test_lsst_image" + file_lab + ".fits",
        filter_name=filter_name_ground,
        out_image=galsim.Image(ngrid_lsst, ngrid_lsst))
    lsst.plot_image("../TestData/test_lsst_image" + file_lab + ".png",
        ngrid=ngrid_lsst,
        filter_name=filter_name_ground, title="LSST " + filter_name_ground)
    # Save the corresponding PSF
    lsst.save_psf("../TestData/test_lsst_psf" + file_lab + ".fits",
        ngrid=ngrid_lsst/4, filter_name=filter_name_ground)
    lsst.plot_psf("../TestData/test_lsst_psf" + file_lab + ".png",
        ngrid=ngrid_lsst/4, title="LSST " + filter_name_ground,
        filter_name=filter_name_ground)

    # WFIRST
    print("\n----- WFIRST -----")
    wfirst = GalSimGalaxyModel(
        telescope_name="WFIRST",
        pixel_scale_arcsec=k_telescopes['WFIRST']['pixel_scale'],
        noise=wfirst_noise(82357),
        galaxy_model=galaxy_model,
        primary_diam_meters=galsim.wfirst.diameter,
        filter_names=k_wfirst_filter_names,
        filter_wavelength_scale=1.0, #1.0e3, # convert from micrometers to nanometers
        atmosphere=False)

    print("LSST AB magnitude:   {:5.4f}".format(lsst.get_magnitude(filter_name_ground)))
    print("LSST flux:   {:5.4f}".format(lsst.get_flux(filter_name_ground)))

    print("WFIRST AB magnitude: {:5.4f}".format(wfirst.get_magnitude(filter_name_space)))
    print("WFIRST flux: {:5.4f}".format(wfirst.get_flux(filter_name_space)))

    ngrid_wfirst = np.ceil(ngrid_lsst * lsst.pixel_scale / wfirst.pixel_scale) #128

    # Save the image
    wfirst.save_image("../TestData/test_wfirst_image" + file_lab + ".fits",
        filter_name=filter_name_space, out_image=galsim.Image(ngrid_wfirst, ngrid_wfirst))
    wfirst.plot_image("../TestData/test_wfirst_image" + file_lab + ".png", ngrid=ngrid_wfirst,
        filter_name=filter_name_space, title="WFIRST " + filter_name_space)
    # Save the corresponding PSF
    wfirst.save_psf("../TestData/test_wfirst_psf" + file_lab + ".fits",
        ngrid=ngrid_wfirst/4, filter_name=filter_name_space)
    wfirst.plot_psf("../TestData/test_wfirst_psf" + file_lab + ".png",
        ngrid=ngrid_wfirst/4, title="WFIRST " + filter_name_space,
        filter_name=filter_name_space)

    lsst_data = lsst.get_image(galsim.Image(ngrid_lsst, ngrid_lsst), add_noise=True,
        filter_name=filter_name_ground).array
    wfirst_data = wfirst.get_image(galsim.Image(ngrid_wfirst, ngrid_wfirst), add_noise=True,
        filter_name=filter_name_space).array

    # -------------------------------------------------------------------------
    ### Save a file with joint image data for input to the Roaster
    segfile = os.path.join(os.path.dirname(__file__),
        '../TestData/test_image_data' + file_lab + '.h5')
    print("Writing {}".format(segfile))
    seg = segments.Segments(segfile)

    seg_ndx = 0
    src_catalog = lsst.params
    seg.save_source_catalog(src_catalog, segment_index=seg_ndx)

    dummy_mask = 1.0
    dummy_background = 0.0

    ### Ground data
    seg.save_images([lsst_data], [lsst.noise.getVariance()], [dummy_mask],
        [dummy_background], segment_index=seg_ndx,
        telescope='lsst',
        filter_name=filter_name_ground)
    seg.save_tel_metadata(telescope='lsst',
        primary_diam=lsst.primary_diam_meters,
        pixel_scale_arcsec=lsst.pixel_scale,
        atmosphere=lsst.atmosphere)
    seg.save_psf_images([lsst.get_psf_image(filter_name_ground).array], segment_index=seg_ndx,
        telescope='lsst',
        filter_name=filter_name_ground)
    save_bandpasses_to_segment(seg, lsst, k_lsst_filter_names, "LSST")

    ### Space data
    seg.save_images([wfirst_data], [wfirst.noise.getVariance()], [dummy_mask],
        [dummy_background], segment_index=seg_ndx,
        telescope='wfirst',
        filter_name=filter_name_space)
    seg.save_tel_metadata(telescope='wfirst',
        primary_diam=wfirst.primary_diam_meters,
        pixel_scale_arcsec=wfirst.pixel_scale,
        atmosphere=wfirst.atmosphere)
    seg.save_psf_images([wfirst.get_psf_image(filter_name_space).array], segment_index=seg_ndx,
        telescope='wfirst',
        filter_name=filter_name_space)
    save_bandpasses_to_segment(seg, wfirst, k_wfirst_filter_names, "WFIRST", scale=1)

    # -------------------------------------------------------------------------

def make_blended_test_image(num_sources=3, random_seed=75256611):
    lsst_pixel_scale_arcsec = 0.2

    ellipticities = [0.05, 0.3, 0.16]
    hlrs = [1.8, 1.0, 2.0]
    orientations = np.array([0.1, 0.25, -0.3]) * np.pi

    noise_model = lsst_noise(82357)

    filter_name = 'y'

    ### Setup the 'segment' image that will contain all the galaxies in the blend
    npix_segment = 128
    # segment_pos = galsim.CelectialCoord(ra=90.*galsim.degrees, dec=-10.*galsim.degrees)
    segment_image = galsim.ImageF(npix_segment, npix_segment, scale=lsst_pixel_scale_arcsec)

    ### Define the galaxy positions in the segment (relative to the center of the segment image)
    ### (see galsim demo13.py)
    pos_rng = galsim.UniformDeviate(random_seed)
    x_gal = (0.4 + 0.1 * np.array([pos_rng() for i in xrange(num_sources)])) * npix_segment
    y_gal = (0.4 + 0.1 * np.array([pos_rng() for i in xrange(num_sources)])) * npix_segment

    npix_gal = 100

    for isrcs in xrange(num_sources):
        # ### Draw every source using the full output array
        # b = galsim.BoundsI(1, nx, 1, ny)
        # sub_image = segment_image[b]

        sub_image = galsim.Image(npix_gal, npix_gal, scale=lsst_pixel_scale_arcsec)

        src_model = GalSimGalaxyModel(pixel_scale=lsst_pixel_scale_arcsec,
            noise=lsst_noise(82357),
            galaxy_model="Spergel",
            primary_diam_meters=8.4, atmosphere=True)
        src_model.params[0]["e"] = ellipticities[isrcs]
        src_model.params[0]["beta"] = orientations[isrcs]
        src_model.params[0]["hlr"] = hlrs[isrcs]
        # p = src_model.get_params()

        # src_model.set_params(p)

        gal_image = src_model.get_image(sub_image, filter_name=filter_name)

        ix = int(math.floor(x_gal[isrcs]+0.5))
        iy = int(math.floor(y_gal[isrcs]+0.5))

        # Create a nominal bound for the postage stamp given the integer part of the
        # position.
        sub_bounds = galsim.BoundsI(ix-0.5*npix_gal, ix+0.5*npix_gal-1,
                                    iy-0.5*npix_gal, iy+0.5*npix_gal-1)
        sub_image.setOrigin(galsim.PositionI(sub_bounds.xmin, sub_bounds.ymin))

        # Find the overlapping bounds between the large image and the individual postage
        # stamp.
        bounds = sub_image.bounds & segment_image.bounds

        segment_image[bounds] += sub_image[bounds]

    segment_image.addNoise(noise_model)

    outfile = "../TestData/test_lsst_blended_image.fits"
    print("Saving to {}".format(outfile))
    segment_image.write(outfile)


if __name__ == "__main__":
    make_test_images()
    # make_blended_test_image()

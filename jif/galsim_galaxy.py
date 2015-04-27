#!/usr/bin/env python
# encoding: utf-8
"""
galsim_galaxy.py

Wrapper for GalSim galaxy models to use in MCMC.
"""
import os
import numpy as np
from operator import add
import galsim


k_SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
k_filter_names = 'ugrizy'

k_spergel_paramnames = ['nu', 'hlr', 'e', 'beta']


### Numpy composite object types for the model parameters for galaxy images under different
### modeling assumptions.
k_galparams_type_sersic = [('redshift', '<f8'), ('n', '<f8'), ('hlr', '<f8'), ('e', '<f8'), 
                           ('beta', '<f8')]
k_galparams_type_sersic += [('flux_sed{:d}'.format(i+1), '<f8') for i in xrange(len(k_SED_names))]

k_galparams_type_spergel = [('redshift', '<f8')] + [(p, '<f8') for p in k_spergel_paramnames]
k_galparams_type_spergel += [('flux_sed{:d}'.format(i+1), '<f8') for i in xrange(len(k_SED_names))]

k_galparams_type_bulgedisk = [('redshift', '<f8')]
k_galparams_type_bulgedisk += [('{}_bulge'.format(p), '<f8') for p in k_spergel_paramnames]
k_galparams_type_bulgedisk += [('{}_disk'.format(p), '<f8') for p in k_spergel_paramnames]
k_galparams_type_bulgedisk += [('flux_sed{:d}_bulge'.format(i+1), '<f8') 
    for i in xrange(len(k_SED_names))]
k_galparams_type_bulgedisk += [('flux_sed{:d}_disk'.format(i+1), '<f8') 
    for i in xrange(len(k_SED_names))]



def lsst_noise(random_seed):
    """
    See GalSim/examples/lsst.yaml

    gain: e- / ADU
    read_noise: Variance in ADU^2
    sky_level: ADU / arcsec^2
    """
    rng = galsim.BaseDeviate(random_seed)
    return galsim.CCDNoise(rng, gain=2.1, read_noise=3.4, sky_level=18000)


def wfirst_noise(random_seed):
    """
    From http://wfirst-web.ipac.caltech.edu/wfDepc/visitor/temp1927222740/results.jsp
    """
    rng = galsim.BaseDeviate(random_seed)
    exposure_time_s = 150.
    pixel_scale_arcsec = 0.11
    read_noise_e_rms = 5.
    sky_background = 3.60382E-01 # e-/pix/s
    gain = 2.1 # e- / ADU
    return galsim.CCDNoise(rng, gain=2.1, 
        read_noise=(read_noise_e_rms / gain) ** 2,
        sky_level=sky_background / pixel_scale_arcsec ** 2 * exposure_time_s)


class GalSimGalaxyModel(object):
    """
    Parametric galaxy model from GalSim for MCMC.

    Mimics GalSim examples/demo1.py
    """
    def __init__(self,
                 psf_sigma=0.5, ### Not used
                 pixel_scale=0.2, 
                 noise=None,
                 galaxy_model="Gaussian",
                 wavelength=1.e-6,
                 primary_diam_meters=2.4,
                 atmosphere=False): 
        self.psf_sigma = psf_sigma
        self.pixel_scale = pixel_scale
        # if noise is None:
        #     noise = galsim.GaussianNoise(sigma=30.)
        self.noise = noise
        self.galaxy_model = galaxy_model
        self.wavelength = wavelength
        self.primary_diam_meters = primary_diam_meters
        self.atmosphere = atmosphere

        ### Set GalSim galaxy model parameters
        # self.params = GalSimGalParams(galaxy_model=galaxy_model)
        if galaxy_model == "Sersic":
            self.params = np.core.records.array([(1., 3.4, 1.8, 0.3, np.pi/4, 1.e5, 0., 0., 0.)],
                dtype=k_galparams_type_sersic)
            self.paramtypes = k_galparams_type_sersic
            self.paramnames = [p[0] for p in k_galparams_type_sersic]
        elif galaxy_model == "Spergel":
            self.params = np.core.records.array([(1., -0.3, 1.8, 0.3, np.pi/4, 1.e5, 0., 0., 0.)],
                dtype=k_galparams_type_spergel)
            self.paramtypes = k_galparams_type_spergel
            self.paramnames = [p[0] for p in k_galparams_type_spergel]
        elif galaxy_model == "BulgeDisk":
            self.params = np.core.records.array([(1., 
                0.5, 0.6, 0.05, 0.0,
                -0.6, 1.8, 0.3, np.pi/4,
                2.e3, 0., 0., 0.,
                0., 1.e3, 0., 0.)],
                dtype=k_galparams_type_bulgedisk)
            self.paramtypes = k_galparams_type_bulgedisk
            self.paramnames = [p[0] for p in k_galparams_type_bulgedisk]
        else:
            raise AttributeError("Unimplemented galaxy model")
        self.n_params = len(self.paramnames)

        ### Set GalSim SED model parameters
        self._load_sed_files()
        ### Load the filters that can be used to draw galaxy images
        self._load_filter_files()

        self.gsparams = galsim.GSParams(
            folding_threshold=1.e-2, # maximum fractional flux that may be folded around edge of FFT
            maxk_threshold=2.e-2,    # k-values less than this may be excluded off edge of FFT
            xvalue_accuracy=1.e-2,   # approximations in real space aim to be this accurate
            kvalue_accuracy=1.e-2,   # approximations in fourier space aim to be this accurate
            shoot_accuracy=1.e-2,    # approximations in photon shooting aim to be this accurate
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

    def _load_filter_files(self):
        """
        Load filters for drawing chromatic objects.

        Copied from GalSim demo12.py
        """
        path, filename = os.path.split(__file__)
        datapath = os.path.abspath(os.path.join(path, "../input/"))
        self.filters = {}
        for filter_name in k_filter_names:
            filter_filename = os.path.join(datapath, 'LSST_{0}.dat'.format(filter_name))
            self.filters[filter_name] = galsim.Bandpass(filter_filename)
            self.filters[filter_name] = self.filters[filter_name].thin(rel_err=1e-4)
        return None

    def set_params(self, p):
        """
        Take a list of parameters and set local variables.

        For use in emcee.
        """
        self.params = np.core.records.array(p, dtype=self.paramtypes)
        return None

    def get_params(self):
        """
        Return a list of model parameter values.
        """
        return self.params.view('<f8')

    def get_psf(self):
        lam_over_diam = self.wavelength / self.primary_diam_meters
        lam_over_diam *= 206265. # arcsec
        optics = galsim.Airy(lam_over_diam, obscuration=0.548, flux=1., gsparams=self.gsparams)
        if self.atmosphere:
            atmos = galsim.Kolmogorov(lam_over_r0=9.e-8, gsparams=self.gsparams)
            psf = galsim.Convolve([atmos, optics])
        else:
            psf = optics
        return psf

    def get_SED(self, gal_comp='', flux_ref_wavelength=500):
        """
        Get the GalSim SED object given the SED parameters and redshift.
        """
        if len(gal_comp) > 0:
            gal_comp = '_' + gal_comp
        SEDs = [self.SEDs[SED_name].withFluxDensity(
            target_flux_density=self.params[0]['flux_sed{:d}{}'.format(i+1, gal_comp)],
            wavelength=flux_ref_wavelength).atRedshift(self.params[0].redshift)
                for i, SED_name in enumerate(self.SEDs)]
        return reduce(add, SEDs)

    def get_image(self, out_image=None, add_noise=False, filter_name='r'):
        if self.galaxy_model == "Gaussian":
            # gal = galsim.Gaussian(flux=self.params.gal_flux, sigma=self.params.gal_sigma)
            # gal_shape = galsim.Shear(g=self.params.e, beta=self.params.beta*galsim.radians)
            # gal = gal.shear(gal_shape)
            raise AttributeError("Unimplemented galaxy model")

        elif self.galaxy_model == "Spergel":
            mono_gal = galsim.Spergel(nu=self.params[0].nu, half_light_radius=self.params[0].hlr,
                # flux=self.params[0].gal_flux, 
                gsparams=self.gsparams)
            SED = self.get_SED()
            gal = galsim.Chromatic(mono_gal, SED)
            gal_shape = galsim.Shear(g=self.params[0].e, beta=self.params[0].beta*galsim.radians)
            gal = gal.shear(gal_shape)

        elif self.galaxy_model == "Sersic":
            mono_gal = galsim.Sersic(n=self.params[0].n, half_light_radius=self.params[0].hlr,
                # flux=self.params[0].gal_flux, 
                gsparams=self.gsparams)
            SED = self.get_SED()
            gal = galsim.Chromatic(mono_gal, SED)
            gal_shape = galsim.Shear(g=self.params[0].e, beta=self.params[0].beta*galsim.radians)
            gal = gal.shear(gal_shape)            

        elif self.galaxy_model == "BulgeDisk":
            mono_bulge = galsim.Spergel(nu=self.params[0].nu_bulge, 
                half_light_radius=self.params[0].hlr_bulge,
                gsparams=self.gsparams)
            SED_bulge = self.get_SED(gal_comp='bulge')
            bulge = galsim.Chromatic(mono_bulge, SED_bulge)
            bulge_shape = galsim.Shear(g=self.params[0].e_bulge, 
                beta=self.params[0].beta_bulge*galsim.radians)
            bulge = bulge.shear(bulge_shape)

            mono_disk = galsim.Spergel(nu=self.params[0].nu_disk, 
                half_light_radius=self.params[0].hlr_disk,
                gsparams=self.gsparams)
            SED_disk = self.get_SED(gal_comp='disk')
            disk = galsim.Chromatic(mono_disk, SED_disk)
            disk_shape = galsim.Shear(g=self.params[0].e_disk, 
                beta=self.params[0].beta_disk*galsim.radians)
            disk = disk.shear(disk_shape)            

            # gal = self.params[0].bulge_frac * bulge + (1 - self.params[0].bulge_frac) * disk
            gal = bulge + disk

        else:
            raise AttributeError("Unimplemented galaxy model")
        final = galsim.Convolve([gal, self.get_psf()])
        # wcs = galsim.PixelScale(self.pixel_scale)'
        try:
            image = final.drawImage(self.filters[filter_name], 
                image=out_image, scale=self.pixel_scale)
            if add_noise:
                if self.noise is not None:
                    image.addNoise(self.noise)
                else:
                    raise AttributeError("A GalSim noise model must be specified to add noise to an\
                        image.")
        except RuntimeError:
            print "Trying to make an image that's too big."
            image = None                    
        return image

    def save_image(self, file_name, filter_name='r'):
        image = self.get_image(filter_name=filter_name)
        image.write(file_name)
        return None

    def plot_image(self, file_name, ngrid=None, filter_name='r'):
        import matplotlib.pyplot as plt
        if ngrid is not None:
            out_image = galsim.Image(ngrid, ngrid)
        else:
            out_image = None
        ###
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(1,1,1)
        im = ax.matshow(self.get_image(out_image, add_noise=True, filter_name=filter_name).array, 
            cmap=plt.get_cmap('coolwarm')) #, vmin=-350, vmax=350)
        fig.colorbar(im)
        fig.savefig(file_name)
        return None

    def get_moments(self, add_noise=True):
        results = self.get_image(add_noise=add_noise).FindAdaptiveMom()
        print 'HSM reports that the image has observed shape and size:'
        print '    e1 = %.3f, e2 = %.3f, sigma = %.3f (pixels)' % (results.observed_shape.e1,
                    results.observed_shape.e2, results.moments_sigma)


def make_test_images():
    """
    Use the GalSimGalaxyModel class to make test images of a galaxy for LSST and WFIRST.
    """
    import os
    import h5py

    print "Making test images for LSST and WFIRST"
    lsst = GalSimGalaxyModel(pixel_scale=0.2, noise=lsst_noise(82357),
        galaxy_model="BulgeDisk",
        wavelength=500.e-9, primary_diam_meters=8.4, atmosphere=True)
    lsst.save_image("../TestData/test_lsst_image.fits", filter_name='r')
    lsst.plot_image("../TestData/test_lsst_image.png", ngrid=64, filter_name='r')

    wfirst = GalSimGalaxyModel(pixel_scale=0.11, noise=wfirst_noise(82357),
        galaxy_model="BulgeDisk",
        wavelength=1.e-6, primary_diam_meters=2.4, atmosphere=False)
    wfirst.save_image("../TestData/test_wfirst_image.fits", filter_name='y')
    wfirst.plot_image("../TestData/test_wfirst_image.png", ngrid=64, filter_name='y')

    lsst_data = lsst.get_image(galsim.Image(64, 64), add_noise=True, filter_name='r').array
    wfirst_data = wfirst.get_image(galsim.Image(64, 64), add_noise=True, filter_name='y').array

    # -------------------------------------------------------------------------
    ### Save a file with joint image data for input to the Roaster
    f = h5py.File(os.path.join(os.path.dirname(__file__), '../TestData/test_image_data.h5'), 'w')
    
    # Define the (sub)groups
    g = f.create_group('ground')
    g_obs = f.create_group('ground/observation')
    g_obs_sex = f.create_group('ground/observation/sextractor')
    g_obs_sex_seg = f.create_group('ground/observation/sextractor/segments')
    
    s = f.create_group('space')
    s_obs = f.create_group('space/observation')
    s_obs_sex = f.create_group('space/observation/sextractor')
    s_obs_sex_seg = f.create_group('space/observation/sextractor/segments')
    
    
    f.attrs['num_sources'] = 1 ### Assert a fixed number of sources for all epochs

    ### Instrument/epoch 1
    g_obs_sex_seg_i = f.create_group("ground/observation/sextractor/segments/0")
    g_obs_sex_seg_i.create_dataset('image', data=lsst_data)
    ### TODO: Add object property data like that that might come out of DMstack or sextractor
    # currently a hack to allow roaster to determine number of objects the
    # same as is done for the data processed by sheller
    g_obs_sex_seg_i.create_dataset('stamp_objprops', data=np.arange(1))    
    ### TODO: Add segmentation mask
    # the real data will create a dataset that is an image of the noise
    # for the galsim_galaxy only a single value characterizing the 
    # variance is generated
    g_obs_sex_seg_i_noise = g_obs_sex_seg_i.create_dataset('noise', data=lsst.noise.getVariance())
    # for consistency with real data, also assign this to the noise dataset
    # attribute
    g_obs_sex_seg_i_noise.attrs['variance'] = lsst.noise.getVariance()
    ### TODO: add WCS information
    ### TODO: add background model(s)
    g.attrs['telescope'] = 'LSST'
    g.attrs['pixel_scale'] = 0.2
    g_obs.attrs['filter_central'] = 500.e-9
    g.attrs['primary_diam'] = 8.4
    g.attrs['atmosphere'] = True
    

    ### Instrument/epoch 2
    s_obs_sex_seg_i = f.create_group("space/observation/sextractor/segments/0")
    s_obs_sex_seg_i.create_dataset('image', data=wfirst_data)
    ### TODO: Add object property data like that that might come out of DMstack or sextractor
    # currently a hack to allow roaster to determine number of objects the
    # same as is done for the data processed by sheller
    s_obs_sex_seg_i.create_dataset('stamp_objprops', data=np.arange(1))
    ### TODO: Add segmentation mask
    # the real data will create a dataset that is an image of the noise
    # for the galsim_galaxy only a single value characterizing the 
    # variance is generated
    s_obs_sex_seg_i_noise = s_obs_sex_seg_i.create_dataset('noise', data=wfirst.noise.getVariance())
    # for consistency with real data, also assign this to the noise dataset
    # attribute
    s_obs_sex_seg_i_noise.attrs['variance'] = lsst.noise.getVariance()
    ### TODO: add WCS information
    ### TODO: add background model(s)
    s.attrs['telescope'] = 'WFIRST'
    s.attrs['pixel_scale'] = 0.11
    s_obs.attrs['filter_central'] = 1.e-6
    s.attrs['primary_diam'] = 2.4
    s.attrs['atmosphere'] = False

    f.close()
    # -------------------------------------------------------------------------    


if __name__ == "__main__":
    make_test_images()


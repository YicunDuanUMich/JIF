#!/usr/bin/env python
# encoding: utf-8
"""
Utility for creating and parsing 'segment' files

A segment file must include the following information:

    - Segment image data in [group_name]/observation/[algorithm]/segments
    - Bandpass information in [group_name]/filters/[filter_name]

Optional additional information might include:
"""
import numpy as np
import h5py


def create_group(f, group_name):
    """
    Create an HDF5 group in f if it does not already exist,
    otherwise just get a reference to it.
    """
    if group_name not in f:
        g = f.create_group(group_name)
    else:
        g = f[group_name]
    return g


class Segments(object):
    """
    I/O for galaxy image segments
    """
    def __init__(self, segment_file):
        self.segment_file = segment_file

        self.file = h5py.File(segment_file, 'w')

    def save_tel_metadata(self, telescope='lsst',
                          primary_diam=8.4, pixel_scale_arcsec=0.2,
                          atmosphere=True):
        g = create_group(self.file, telescope)
        g.attrs['telescope'] = telescope
        g.attrs['primary_diam'] = primary_diam
        g.attrs['pixel_scale_arcsec'] = pixel_scale_arcsec
        g.attrs['atmosphere'] = atmosphere
        return None

    def save_wcs(self):
        raise NotImplementedError()

    def save_backgrounds(self):
        raise NotImplementedError()

    def save_images(self,
                    image_list,
                    noise_list,
                    mask_list,
                    background_list,
                    telescope = 'lsst',
                    filter_name = 'r'):
        """
        Save images for the segments from a single telescope
        """
        segment_name = 'segment/{}/band_{}'.format(telescope, filter_name)
        for iseg, im in enumerate(image_list):
            seg = create_group(self.file, segment_name + '/epoch_{:d}'.format(iseg))
            # image - background
            seg.create_dataset('image', data=im)
            # rms noise
            seg.create_dataset('noise', data=noise_list[iseg])
            # estimate the variance of this noise image and save as attribute
            seg.attrs['variance'] = np.var(noise_list[iseg])
            seg.create_dataset('segmask', data=mask_list[iseg])
            seg.create_dataset('background', data=background_list[iseg])
        return None

    def save_psf_images(self,
                        image_list,
                        telescope='lsst',
                        filter_name='r',
                        model_names=None):
        """
        Save an image of the estimated PSF for each segment

        The elements of 'image_list' do not have to be images. In this case,
        specify how to parse the 'image' replacements with a list of descriptive
        strings in 'model_names'.
        """
        segment_name = 'segment/{}/band_{}'.format(telescope, filter_name)
        for iseg, im in enumerate(image_list):
            seg = create_group(self.file, segment_name + '/epoch_{:d}'.format(iseg))
            seg.create_dataset('psf', data=im)
            if model_names is None:
                seg.attrs['psf_type'] = 'image'
            else: ### Assume a list of names of PSF model types
                seg.attrs['psf_type'] = model_names[iseg]
        return None

    def save_bandpasses(self, filters_list, waves_nm_list, throughputs_list,
                        telescope='lsst'):
        """
        Save bandpasses for a single telescope as lookup tables.
        """
        for i, filter_name in enumerate(filters_list):
            bp = self.file.create_group('{}/filters/{}'.format(telescope,
                filter_name))
            bp.create_dataset('waves_nm', data=waves_nm_list[i])
            bp.create_dataset('throughput', data=throughputs_list[i])
        return None

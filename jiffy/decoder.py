import footprints
import galsim
from pathlib import Path
from jiffy.galsim_galaxy import GalsimGalaxyModel
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import yaml


class PriorDistribution:
    def set_seed(self, seed):
        self.random_state_generator = random.Random(seed)

    def sample(self, n_samples):
        raise NotImplementedError()
    
class DC2PriorDistribution(PriorDistribution):
    FLUX_SCALE = 10
    FLUX_MIN = 1
    DR_SCALE = 3
    DR_MIN = 0.5
    def __init__(self, path_to_bayesian_model):
        super().__init__()
        with open(path_to_bayesian_model, "rb") as f:
            self.prior = pickle.load(f)  # this is a BayesianGaussianMixture in sklearn

    def sample(self, n_samples):
        self.prior.random_state = self.random_state_generator.randint(1, 1e7)
        # (log(hlr), log(flux + 1), sqrt(dx ** 2 + dy ** 2), sqrt(e1 ** 2, e2 ** 2))
        samples = self.prior.sample(n_samples)[0]
        log_hlr = samples[:, 0].reshape((-1, 1))
        de = samples[:, -1].reshape((-1, 1))
        log_flux = np.log(np.random.random_sample(n_samples) * self.FLUX_SCALE + self.FLUX_MIN).reshape((-1, 1))
        dr = (np.random.random_sample(n_samples) * self.DR_SCALE + self.DR_MIN).reshape((-1, 1))

        return np.hstack((log_hlr, log_flux, dr, de))


class Decoder:
    TRUTH_PARAMS = ["nu", "hlr", "flux", "dx", "dy", "e1", "e2"]
    def __init__(self, 
                 prior_dist: PriorDistribution,
                 image_infer_config_template,
                 image_slen = 80,
                 noise_var = 1e-8,
                 mask_default = 1.0,
                 bg_default = 0.0,
                 output_dir = "./decoder_output",
                 seed = 7272) -> None:
            self.prior_dist = prior_dist
            self.prior_dist.set_seed(seed)
            self.image_infer_config_template = image_infer_config_template
            self.image_slen = image_slen
            self.noise_var = noise_var
            self.mask_default = mask_default
            self.bg_default = bg_default
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
            self.random_angle_generator = random.Random(seed)

            self.ggm = GalsimGalaxyModel({"model": {"psf_class": "GalsimPSFModel"}},
                                         active_parameters=[])

    def sample_one_image(self, n_samples):
        prior_params = self.prior_dist.sample(n_samples)
        truth_params = []

        image = None
        for sample_i in range(n_samples):
            log_hlr, log_flux, dr, e = prior_params[sample_i, :]
            hlr, flux = np.exp(log_hlr), np.exp(log_flux)
            self.ggm.set_param_by_name("hlr", hlr)
            self.ggm.set_param_by_name("flux", flux)
            random_angle_1 = self.random_angle_generator.randint(0, 45) / 180 * np.pi
            random_angle_2 = self.random_angle_generator.randint(0, 359) / 180 * np.pi
            e1, e2 = e * np.cos(random_angle_1), e * np.sin(random_angle_1)
            dx, dy = dr * np.cos(random_angle_2), dr * np.sin(random_angle_2)
            self.ggm.set_param_by_name("e1", e1)
            self.ggm.set_param_by_name("e2", e2)
            self.ggm.set_param_by_name("dx", dx)
            self.ggm.set_param_by_name("dy", dy)
            assert self.ggm.validate_params()
            
            image = self.ggm.get_image(ngrid_x=self.image_slen, 
                                       ngrid_y=self.image_slen,
                                       image=image)
            assert image is not None
            
            truth_params.append([self.ggm.params["nu"][0], hlr, flux, dx, dy, e1, e2])
        
        noise = galsim.GaussianNoise(sigma=np.sqrt(self.noise_var))
        image.addNoise(noise)
        
        return pd.DataFrame(truth_params, columns=self.TRUTH_PARAMS), image
    
    def sample_one_footprint(self, n_samples, image_index):
        image_folder_path = self.output_dir / f"image_{image_index:04d}"
        image_folder_path.mkdir(exist_ok=True)
        image_file_name = f"jiffy_decoder_image_{image_index:04d}"
        truth_params, image = self.sample_one_image(n_samples)

        truth_params.to_csv(image_folder_path / (image_file_name + ".csv"),
                            index=False)

        # galsim.fits.write(image, 
        #                   (self.output_dir / (image_file_name + ".fits")).as_posix())
        
        footprint_file = footprints.Footprints(image_folder_path / (image_file_name + ".h5"))
        footprint_file.save_images([image.array], [self.noise_var], 
                                   [self.mask_default], [self.bg_default],
                                   segment_index=0, telescope="LSST",
                                   filter_name="r")
        footprint_file.save_tel_metadata()

        fig, ax = plt.subplots(figsize=(6, 6))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt_im = ax.imshow(image.array, cmap="bone")
        fig.colorbar(plt_im, cax=cax)
        fig.savefig(image_folder_path / (image_file_name + ".pdf"),
                    bbox_inches="tight",
                    dpi=800)
        plt.close()
        
        # create config file for inference
        self.image_infer_config_template["io"]["infile"] = f"./decoder_output/image_{image_index:04d}/jiffy_decoder_image_{image_index:04d}.h5"
        self.image_infer_config_template["io"]["roaster_outfile"] = f"./encoder_output/jiffy_encoder_out_{image_index:04d}"
        self.image_infer_config_template["model"]["num_sources"] = n_samples
        self.image_infer_config_template["init"]["init_param_file"] = f"../config/n_{n_samples}_samples_init_params.cfg"

        infer_config_file_path = image_folder_path / f"infer_image_{image_index:04d}.yaml"
        with open(infer_config_file_path, "w") as f:
            yaml.dump(self.image_infer_config_template, f)

        return infer_config_file_path

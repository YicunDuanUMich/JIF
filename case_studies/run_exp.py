from jiffy import roaster, roaster_inspector
from jiffy.decoder import Decoder, DC2PriorDistribution
import yaml
from jiffy.roaster_args import RoasterArgs
from jiffy.roaster_inspector_args import RoasterInspectorArgs
from multiprocessing import Pool
import os


if __name__ == "__main__":
    print(os.getcwd())
    os.chdir("/home/pduan/jif_related/JIF/case_studies/")
    print(os.getcwd())
    with open("../config/base_infer.yaml", "r") as f:
        base_infer_config = yaml.safe_load(f)

    decoder = Decoder(prior_dist=DC2PriorDistribution("../jiffy/gmfile_disk.pkl"),
                  image_infer_config_template=base_infer_config)
    
    n_images = 100
    n_samples_per_image = 2

    infer_config_file_paths = []
    for i_image in range(n_images):
        print(f"generating image [{i_image}]")
        infer_config_file_path = decoder.sample_one_footprint(n_samples=n_samples_per_image, image_index=i_image)
        infer_config_file_paths.append(infer_config_file_path)

    roaster_args_list = [RoasterArgs(config_file=infer_file, parallize=False) 
                         for infer_file in infer_config_file_paths]

    def run_roaster_func(roaster_args):
        cur_roaster = roaster.init_roaster(roaster_args)
        roaster.do_sampling(roaster_args, cur_roaster)

    with Pool(processes=32) as pool:
        pool.map(run_roaster_func, roaster_args_list)


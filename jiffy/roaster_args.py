class RoasterArgs:
    def __init__(self, config_file, parallize) -> None:
        self.config_file = config_file
        self.footprint_number = 0
        self.unparallelize = False if parallize else True
        self.verbose = True
        self.show_progress_bar = False
        self.initialize_from_image = False
        self.cluster_walkers = True
        self.cluster_walkers_thresh = 4
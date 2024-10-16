class RoasterInspectorArgs:
    def __init__(self, infile, roaster_config, keeplast = 0) -> None:
        self.infile = infile
        self.roaster_config = roaster_config
        self.keeplast = keeplast
        self.verbose = True
        self.footprint_number = 0
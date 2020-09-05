from . import lib

photoz = lib.libbcnz.photoz_wrapper

# Helper functions.
flatten = lib.utils.flatten
load_models = lib.utils.load_models

# Newer imports. All above should later be deleted.
from . import calib
from . import config
from . import data
from . import model

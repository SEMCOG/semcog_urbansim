import orca
import shutil

import os

import models
import utils
from urbansim.utils import misc, networks
import output_indicators

data_out = './runs/run1029.h5'
output_indicators.main(
    data_out,
    2020,
    2050,
    spacing=5,
    upload_to_carto=False,
)
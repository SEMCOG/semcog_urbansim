import orca
import shutil

import os
orca.add_injectable('use_checkpoint', False)
# hlcm configs
orca.add_injectable('hlcm_model_path', '/mnt/hgfs/RDF2050/estimation/models/models_23dec06')
orca.add_injectable('yaml_configs', 'yaml_configs_nn.yaml')

import models
import utils
from urbansim.utils import misc, networks
import output_indicators

# set up run
base_year = 2020
final_year = 2050
indicator_spacing = 5
upload_to_carto = True
run_debug = False
add_2019 = True
# data_out = './runs/run2120_TAZ_refinement_090823_school_legis.h5'
data_out = 'runs/run1241.h5'

output_indicators.main(
    data_out,
    base_year,
    final_year,
    spacing=indicator_spacing,
    upload_to_carto=upload_to_carto,
    add_2019=add_2019,
)
import orca
import shutil

import os

import models, utils
from urbansim.utils import misc, networks
import output_indicators

with open(os.path.join(os.getenv('DATA_HOME', "."), 'RUNNUM'), 'r') as f:
    start_num = int(f.read())

for i in xrange(start_num, 1, -1):
    data_out = os.path.join(misc.runs_dir(), "run%d.h5" % i)
    if os.path.isfile(data_out):
        print "runnum:", i
        break

data_out = './runs/run4032_hu_hh_base_2_gq_jobs_v5_hhpop_agegrp.h5'
output_indicators.main(data_out)

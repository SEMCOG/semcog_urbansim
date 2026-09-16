import os
import sys

import shutil
from urbansim.utils import misc, networks

SUMMARY_EXPORT_PATH = "/home/da/share/urbansim/RDF2050/model_runs"

with open(os.path.join(os.getenv('DATA_HOME', "."), 'RUNNUM'), 'r') as f:
    start_num = int(f.read())

for i in range(start_num, 1, -1):
    run_name = os.path.join(misc.runs_dir(), "run%d.h5" % i)
    if os.path.isfile(run_name):
        print("runnum:", i)
        break

outdir = run_name.replace('.h5', '')
folder_name = outdir.split('/')[-1]
if not (os.path.exists(outdir)):
	print("Summary tables folder not found, aborting..")
	sys.exit()

# this line is running really slow, consider other options
shutil.copytree(outdir, os.path.join(SUMMARY_EXPORT_PATH, folder_name))

print('done.')
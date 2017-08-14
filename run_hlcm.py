import os
import sys
import time
import subprocess

python = sys.executable
root_path = os.path.dirname(__file__)


def run(filename, block=False, args=None):
    """"Run Python file relative to script with or without blocking."""
    path = os.path.join(root_path, filename)
    command = [python, path]
    if args:
        command = command + args

    if block:
        return subprocess.check_call(command)
    else:
        return subprocess.Popen(command)

qlids = [400125, 100005, 300125, 300003, 200125, 200003, 400003, 100003,
       300099, 200099, 100125, 400099, 100099, 200005, 400161, 300005,
       300161, 200161, 100161, 400093, 300093, 200147, 300147, 300115,
       400005, 100147, 200115, 200093, 400115, 400147, 100115, 100093]

#for income_quartile_id in range(1,5):
for income_quartile_id in qlids:
    run('autospec_distributed_manager.py', args = ['-o',  'localhost', 
                                                       '-p', '6379',
                                                       '-d', 'expvars.h5',
                                                       '-t', 'buildings',
                                                       '-c', 'households',
                                                       '-m', 'hlcm',
                                                       '-f', 'building_id>1',
                                                       '-a', 'residential_units>0',
                                                       '-s', 'qlid',
                                                       '-i', str(income_quartile_id),
                                                       '-fs', 'recipe'])   ###NOTE THIS ARG-  USE OR NOT IF RECIPE

    time.sleep(12)

    run('autospec_distributed_worker.py', block=True, args = ['-o',  'localhost',
                                                       '-p', '6379',
                                                       '-d', 'expvars.h5',
                                                       '-t', 'parcel',
                                                       '-k'])
                                                       #'-s'])   ### NOTE THIS ARG-  ONLY IF YOU WANT TO EXPORT RESULTS

    time.sleep(10)

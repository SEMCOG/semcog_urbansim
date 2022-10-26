import orca
import os
import yaml
from urbansim_templates.models import LargeMultinomialLogitStep
from urbansim_templates import modelmanager as mm
mm.initialize('configs/elcm_2050')


def generate_yaml_configs():
    hlcm_yaml = os.listdir('configs/hlcm_2050')
    hlcm_yaml = ["hlcm_2050/"+path for path in hlcm_yaml if '.yaml' in path]

    elcm_yaml = os.listdir('configs/elcm_2050')
    elcm_yaml = ["elcm_2050/"+path for path in elcm_yaml if '.yaml' in path]
    obj = {
        'hlcm': hlcm_yaml,
        'elcm': elcm_yaml
    }
    with open("./configs/yaml_configs_2050.yaml", 'w') as f:
        yaml.dump(obj, f, default_flow_style=False)

if __name__ == "__main__":
    generate_yaml_configs()
    import models
    orca.add_injectable('year', 2020)
    orca.run(["build_networks_2050"])
    orca.run(["neighborhood_vars"])
    # set year to 2050 
    orca.add_injectable('year', 2030)
    orca.run(["mcd_hu_sampling"])
    # orca.run(['elcm_800003'])
    orca.run(['hlcm_125'])
    orca.run(['nonres_repm11'])
    print('done')
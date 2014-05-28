from urbansim.developer import sqftproforma
from urbansim.utils import networks
from urbansim.models.yamlmodelrunner import *
from dataset import *


# residential sales hedonic
def rsh_estimate(dset):
    df = dset.merge_nodes(HomeSales(dset).build_df())
    return hedonic_estimate(df, "rsh.yaml")


def rsh_simulate(dset):
    df = dset.merge_nodes(Buildings(dset, addprices=False).build_df())
    return hedonic_simulate(df, "rsh.yaml", dset.buildings, "residential_sales_price")


# residential rent hedonic
def rrh_estimate(dset):
    df = dset.merge_nodes(Apartments(dset).build_df())
    return hedonic_estimate(df, "rrh.yaml")


def rrh_simulate(dset):
    df = dset.merge_nodes(Buildings(dset, addprices=False).build_df())
    return hedonic_simulate(df, "rrh.yaml", dset.buildings, "residential_rent")


# non-residential hedonic
def nrh_estimate(dset):
    df = dset.merge_nodes(CoStar(dset).build_df())
    return hedonic_estimate(df, "nrh.yaml")


def nrh_simulate(dset):
    df = dset.merge_nodes(Buildings(dset, addprices=False).build_df())
    return hedonic_simulate(df, "nrh.yaml", dset.buildings, "non_residential_rent")


# household location choice
def hlcm_estimate(dset):
    households = dset.households
    buildings = dset.buildings
    return lcm_estimate(households, "building_id", buildings, "hlcm.yaml")


def hlcm_simulate(dset):
    households = dset.households
    buildings = dset.buildings
    units = get_vacant_units(households, "building_id", buildings, "residential_units")
    return lcm_simulate(households, units, "hlcm.yaml", dset.households, "building_id")


# employment location choice
def elcm_estimate(dset):
    jobs = dset.jobs_for_estimation
    buildings = dset.buildings
    return lcm_estimate(jobs, "building_id", buildings, "elcm.yaml")


def elcm_simulate(dset):
    jobs = dset.jobs
    buildings = dset.buildings
    units = get_vacant_units(jobs, "building_id", buildings, "job_spaces")
    return lcm_simulate(jobs, units, "elcm.yaml", dset.jobs, "building_id")


def households_relocation(dset):
    return simple_relocation(dset.households, .05)


def jobs_relocation(dset):
    return simple_relocation(dset.jobs, .08)


def households_transition(dset):
    return simple_transition(dset, "households", .05)


def jobs_transition(dset):
    return simple_transition(dset, "jobs", .05)


def build_networks():
    if not networks.NETWORKS:
        networks.NETWORKS = networks.Networks(
            [os.path.join(misc.data_dir(), x) for x in ['osm_bayarea.jar']],
            factors=[1.0],
            maxdistances=[2000],
            twoway=[1],
            impedances=None)


def _update_xys(dset):
    for dfname in ["households", "jobs"]:
        print "Adding xys to dataframe: %s" % dfname
        df = dset.add_xy(dset.fetch(dfname))
        dset.save_tmptbl(dfname, df)


def neighborhood_vars(dset):
    _update_xys(dset)
    nodes = networks.from_yaml(dset, "networks.yaml")
    dset.save_tmptbl("nodes", nodes)


def price_vars(dset):
    nodes = networks.from_yaml(dset, "networks2.yaml")
    dset.save_tmptbl("nodes_prices", nodes)


def feasibility(dset):
    developer = sqftproforma.SqFtProForma()

    parcels = Parcels(dset)
    df = parcels.build_df()

    # add prices for each use
    for t in developer.config.uses:
        df[t] = parcels.price(t)

    # convert from cost to yearly rent
    df["residential"] *= developer.config.cap_rate

    far_predictions = {}
    for form in developer.config.forms:
        print form
        tmp = df[parcels.allowed(form)]
        print tmp.describe()
        far_predictions[form] = developer.lookup(form, tmp)

    print far_predictions
    far_predictions = pd.DataFrame(far_predictions)
    print far_predictions

    dset.save_tmptbl("feasibility", far_predictions)


from urbansim.models import RegressionModel, SegmentedRegressionModel, \
    MNLLocationChoiceModel, SegmentedMNLLocationChoiceModel, \
    GrowthRateTransition
from urbansim.developer import sqftproforma, developer
import numpy as np
import pandas as pd
import urbansim.sim.simulation as sim
from urbansim.utils import misc
import os


def get_run_filename():
    return os.path.join(misc.runs_dir(), "run%d.h5" % misc.get_run_number())


def change_store(store_name):
    sim.add_injectable("store",
                       pd.HDFStore(os.path.join(misc.data_dir(),
                                                store_name), mode="r"))


def change_scenario(scenario):
    assert scenario in sim.get_injectable("scenario_inputs"), \
        "Invalid scenario name"
    print "Changing scenario to '%s'" % scenario
    sim.add_injectable("scenario", scenario)


def conditional_upzone(scenario, attr_name, upzone_name):
    scenario_inputs = sim.get_injectable("scenario_inputs")
    zoning_baseline = sim.get_table(
        scenario_inputs["baseline"]["zoning_table_name"])
    attr = zoning_baseline[attr_name]
    if scenario != "baseline":
        zoning_scenario = sim.get_table(
            scenario_inputs[scenario]["zoning_table_name"])
        upzone = zoning_scenario[upzone_name].dropna()
        attr = pd.concat([attr, upzone], axis=1).max(skipna=True, axis=1)
    return attr


def enable_logging():
    from urbansim.utils import logutil
    logutil.set_log_level(logutil.logging.INFO)
    logutil.log_to_stream()


def deal_with_nas(df):
    lenbefore = len(df)
    df = df.dropna(how='any')
    lenafter = len(df)
    if lenafter != lenbefore:
        print "Dropped %d rows because they contained nans" % \
              (lenbefore-lenafter)
    return df


def to_frame(tables, cfg, additional_columns=[]):
    cfg = yaml_to_class(cfg).from_yaml(str_or_buffer=cfg)
    tables = [t for t in tables if t is not None]
    columns = misc.column_list(tables, cfg.columns_used()) + additional_columns
    if len(tables) > 1:
        df = sim.merge_tables(target=tables[0].name,
                              tables=tables, columns=columns)
    else:
        df = tables[0].to_frame(columns)

    df = deal_with_nas(df)
    return df


def yaml_to_class(cfg):
    import yaml
    model_type = yaml.load(open(cfg))["model_type"]
    return {
        "regression": RegressionModel,
        "segmented_regression": SegmentedRegressionModel,
        "locationchoice": MNLLocationChoiceModel,
        "segmented_locationchoice": SegmentedMNLLocationChoiceModel
    }[model_type]


def hedonic_estimate(cfg, tbl, nodes):
    cfg = misc.config(cfg)
    df = to_frame([tbl, nodes], cfg)
    return yaml_to_class(cfg).fit_from_cfg(df, cfg)


def hedonic_simulate(cfg, tbl, nodes, out_fname):
    cfg = misc.config(cfg)
    df = to_frame([tbl, nodes], cfg)
    price_or_rent, _ = yaml_to_class(cfg).predict_from_cfg(df, cfg)
    tbl.update_col_from_series(out_fname, price_or_rent)


def lcm_estimate(cfg, choosers, chosen_fname, buildings, nodes):
    cfg = misc.config(cfg)
    choosers = to_frame([choosers], cfg, additional_columns=[chosen_fname])
    alternatives = to_frame([buildings, nodes], cfg)
    return yaml_to_class(cfg).fit_from_cfg(choosers,
                                           chosen_fname,
                                           alternatives,
                                           cfg)


def lcm_simulate(cfg, choosers, buildings, nodes, out_fname,
                 supply_fname, vacant_fname):
    """
    Simulate the location choices for the specified choosers

    Parameters
    ----------
    cfg : string
        The name of the yaml config file from which to read the location
        choice model.
    choosers : DataFrame
        A dataframe of agents doing the choosing.
    buildings : DataFrame
        A dataframe of buildings which the choosers are locating in and which
        have a supply.
    nodes : DataFrame
        A land use dataset to give neighborhood info around the buildings -
        will be joined to the buildings.
    out_dfname : string
        The name of the dataframe to write the simulated location to.
    out_fname : string
        The column name to write the simulated location to.
    supply_fname : string
        The string in the buildings table that indicates the amount of
        available units there are for choosers, vacant or not.
    vacant_fname : string
        The string in the buildings table that indicates the amount of vacant
        units there will be for choosers.
    """
    cfg = misc.config(cfg)

    choosers_df = to_frame([choosers], cfg, additional_columns=[out_fname])
    locations_df = to_frame([buildings, nodes], cfg,
                            [supply_fname, vacant_fname])

    available_units = locations_df[supply_fname]
    vacant_units = locations_df[vacant_fname]

    print "There are %d total available units" % available_units.sum()
    print "    and %d total choosers" % len(choosers.index)
    print "    but there are %d overfull buildings" % \
          len(vacant_units[vacant_units < 0])

    vacant_units = vacant_units[vacant_units > 0]
    units = locations_df.loc[np.repeat(vacant_units.index,
                             vacant_units.values.astype('int'))].reset_index()

    print "    for a total of %d empty units" % vacant_units.sum()
    print "    in %d buildings total in the region" % len(vacant_units)

    movers = choosers_df[choosers_df[out_fname] == -1]

    new_units, _ = yaml_to_class(cfg).predict_from_cfg(movers, units, cfg)

    # go from units back to buildings
    new_buildings = pd.Series(units.loc[new_units.values][out_fname].values,
                              index=new_units.index)

    choosers.update_col_from_series(out_fname, new_buildings)
    _print_number_unplaced(choosers, out_fname)


def simple_relocation(choosers, relocation_rate, fieldname):
    print "Total agents: %d" % len(choosers)
    _print_number_unplaced(choosers, fieldname)

    print "Assinging for relocation..."
    chooser_ids = np.random.choice(choosers.index, size=int(relocation_rate *
                                   len(choosers)), replace=False)
    choosers.update_col_from_series(fieldname,
                                    pd.Series(-1, index=chooser_ids))

    _print_number_unplaced(choosers, fieldname)


def simple_transition(tbl, rate):
    transition = GrowthRateTransition(rate)
    df = tbl.to_frame(tbl.local_columns)

    print "%d agents before transition" % len(df.index)
    df, added, copied, removed = transition.transition(df, None)
    print "%d agents after transition" % len(df.index)

    sim.add_table(tbl.name, df)


def _print_number_unplaced(df, fieldname):
    print "Total currently unplaced: %d" % \
          df[fieldname].value_counts().get(-1, 0)
          
def run_scheduled_development_events(scheduled_development_events, buildings, year):

    #Use only scheduled development events for the current simulation year
    sched_dev = scheduled_development_events[scheduled_development_events.year_built==year]
    number_of_events = len(sched_dev)
    print 'Inserting %s scheduled development events into the building table' % number_of_events
    
    if number_of_events > 0:
        max_building_id = buildings.index.values.max()
        new_building_ids = np.arange(max_building_id + 1, max_building_id + number_of_events + 1)
        sched_dev['building_id'] = new_building_ids
        sched_dev = sched_dev.set_index('building_id')
        print buildings.local_columns ##Make sure that this is only the primary attributes, otherwise there will be nan's post-merge...
        buildings = buildings.to_frame(buildings.local_columns)
        
        #Using the Developer model's merge function to merge in new buildings
        from urbansim.developer.developer import Developer
        merge = Developer(pd.DataFrame({})).merge
        all_buildings = merge(buildings,sched_dev[buildings.columns])
        
        sim.add_table("buildings", all_buildings)


def run_feasibility(parcels, parcel_price_callback,
                    parcel_use_allowed_callback, residential_to_yearly=True):
    """
    Execute development feasibility on all parcels

    Parameters
    ----------
    parcels : DataFrame Wrapper
        The data frame wrapper for the parcel data
    parcel_price_callback : function
        A callback which takes each use of the pro forma and returns a series
        with index as parcel_id and value as yearly_rent
    parcel_use_allowed_callback : function
        A callback which takes each form of the pro forma and returns a series
        with index as parcel_id and value and boolean whether the form
        is allowed on the parcel
    residential_to_yearly : boolean (default true)
        Whether to use the cap rate to convert the residential price from total
        sales price per sqft to rent per sqft

    Returns
    -------
    Adds a table called feasibility to the sim object (returns nothing)
    """
    pf = sqftproforma.SqFtProForma()

    df = parcels.to_frame()

    # add prices for each use
    for use in pf.config.uses:
        df[use] = parcel_price_callback(use)

    # convert from cost to yearly rent
    if residential_to_yearly:
        df["residential"] *= pf.config.cap_rate

    print "Describe of the yearly rent by use"
    print df[pf.config.uses].describe()
    d = {}
    for form in pf.config.forms:
        print "Computing feasibility for form %s" % form
        d[form] = pf.lookup(form, df[parcel_use_allowed_callback(form)])

    far_predictions = pd.concat(d.values(), keys=d.keys(), axis=1)
    import time
    seconds = time.time()
    far_predictions.to_csv('c://users//janowicz//desktop//feasibility%s.csv'%seconds)
    sim.add_table("feasibility", far_predictions)


def run_developer(forms, agents, buildings, supply_fname, parcel_size,
                  ave_unit_size, total_units, feasibility, year=None,
                  target_vacancy=.1, form_to_btype_callback=None,
                  add_more_columns_callback=None, max_parcel_size=200000,
                  residential=True, bldg_sqft_per_job=400.0):
    """
    Run the developer model to pick and build buildings

    Parameters
    ----------
    forms : string or list of strings
        Passed directly dev.pick
    agents : DataFrame Wrapper
        Used to compute the current demand for units/floorspace in the area
    buildings : DataFrame Wrapper
        Used to compute the current supply of units/floorspace in the area
    supply_fname : string
        Identifies the column in buildings which indicates the supply of
        units/floorspace
    parcel_size : Series
        Passed directly to dev.pick
    ave_unit_size : Series
        Passed directly to dev.pick - average residential unit size
    total_units : Series
        Passed directly to dev.pick - total current residential_units /
        job_spaces
    feasibility : DataFrame Wrapper
        The output from feasibility above (the table called 'feasibility')
    year : int
        The year of the simulation - will be assigned to 'year_built' on the
        new buildings
    target_vacancy : float
        The target vacancy rate - used to determine how much to build
    form_to_btype_callback : function
        Will be used to convert the 'forms' in the pro forma to
        'building_type_id' in the larger model
    add_more_columns_callback : function
        Takes a dataframe and returns a dataframe - is used to make custom
        modifications to the new buildings that get added
    max_parcel_size : float
        Passed directly to dev.pick - max parcel size to consider
    residential : boolean
        Passed directly to dev.pick - switches between adding/computing
        residential_units and job_spaces
    bldg_sqft_per_job : float
        Passed directly to dev.pick - specified the multiplier between
        floor spaces and job spaces for this form (does not vary by parcel
        as ave_unit_size does)

    Returns
    -------
    Writes the result back to the buildings table (returns nothing)
    """

    dev = developer.Developer(feasibility.to_frame())

    target_units = dev.\
        compute_units_to_build(len(agents),
                               buildings[supply_fname].sum(),
                               target_vacancy)

    new_buildings = dev.pick(forms,
                             target_units,
                             parcel_size,
                             ave_unit_size,
                             total_units,
                             max_parcel_size=max_parcel_size,
                             drop_after_build=True,
                             residential=residential,
                             bldg_sqft_per_job=bldg_sqft_per_job)

    if new_buildings is None:
        return

    if year is not None:
        new_buildings["year_built"] = year

    if not isinstance(forms, list):
        # form gets set only if forms is a list
        new_buildings["form"] = forms

    if form_to_btype_callback is not None:
        new_buildings["building_type_id"] = new_buildings["form"].\
            apply(form_to_btype_callback)

    new_buildings["stories"] = new_buildings.stories.apply(np.ceil)

    if add_more_columns_callback is not None:
        new_buildings = add_more_columns_callback(new_buildings)

    print "Adding {} buildings with {:,} {}".\
        format(len(new_buildings),
               new_buildings[supply_fname].sum(),
               supply_fname)

    all_buildings = dev.merge(buildings.to_frame(buildings.local_columns),
                              new_buildings[buildings.local_columns])
    sim.add_table("buildings", all_buildings)

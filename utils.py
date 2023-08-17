import os

import numpy as np
import orca
import pandas as pd
from sklearn.metrics import accuracy_score
from urbansim.models import (
    RegressionModel,
    SegmentedRegressionModel,
    MNLDiscreteChoiceModel,
    SegmentedMNLDiscreteChoiceModel,
    GrowthRateTransition,
)
from urbansim.utils import misc
import numbers
import logging


def get_run_filename():
    return os.path.join(misc.runs_dir(), "run%d.h5" % misc.get_run_number())


def change_store(store_name):
    orca.add_injectable(
        "store", pd.HDFStore(os.path.join(misc.data_dir(), store_name), mode="r")
    )


def change_scenario(scenario):
    assert scenario in orca.get_injectable("scenario_inputs"), "Invalid scenario name"
    print("Changing scenario to '%s'" % scenario)
    orca.add_injectable("scenario", scenario)


def conditional_upzone(scenario, attr_name, upzone_name):
    scenario_inputs = orca.get_injectable("scenario_inputs")
    zoning_baseline = orca.get_table(scenario_inputs["baseline"]["zoning_table_name"])
    attr = zoning_baseline[attr_name]
    if scenario != "baseline":
        zoning_scenario = orca.get_table(scenario_inputs[scenario]["zoning_table_name"])
        upzone = zoning_scenario[upzone_name].dropna()
        attr = pd.concat([attr, upzone], axis=1).max(skipna=True, axis=1)
    return attr


def enable_logging():
    from urbansim.utils import logutil

    logutil.set_log_level(logutil.logging.INFO)
    logutil.log_to_stream()


def deal_with_nas(df):
    df_cnt = len(df)
    fail = False

    df = df.replace([np.inf, -np.inf], np.nan)
    # df[df.isnull().any(axis=1)].to_csv('nulls.csv')
    for col in df.columns:
        s_cnt = df[col].count()
        if df_cnt != s_cnt:
            fail = True
            print(
                "Found %d nas or inf (out of %d) in column %s"
                % (df_cnt - s_cnt, df_cnt, col)
            )

    assert not fail, "NAs were found in dataframe, please fix"
    return df


def fill_nas_from_config(dfname, df):
    df_cnt = len(df)
    fillna_config = orca.get_injectable("fillna_config")
    fillna_config_df = fillna_config[dfname]
    for fname in fillna_config_df:
        filltyp, dtyp = fillna_config_df[fname]
        s_cnt = df[fname].count()
        fill_cnt = df_cnt - s_cnt
        if filltyp == "zero":
            val = 0
        elif filltyp == "mode":
            val = df[fname].dropna().value_counts().idxmax()
        elif filltyp == "median":
            val = df[fname].dropna().quantile()
        else:
            assert 0, "Fill type not found!"
        print(
            "Filling column {} with value {} ({} values)".format(fname, val, fill_cnt)
        )
        df[fname] = df[fname].fillna(val).astype(dtyp)
    return df


def to_frame(tables, cfg, additional_columns=[]):
    cfg = yaml_to_class(cfg).from_yaml(str_or_buffer=cfg)
    tables = [t for t in tables if t is not None]
    columns = misc.column_list(tables, cfg.columns_used()) + additional_columns
    if len(tables) > 1:
        df = orca.merge_tables(target=tables[0].name, tables=tables, columns=columns)
    else:
        df = tables[0].to_frame(columns)
    df = deal_with_nas(df)
    return df


def yaml_to_class(cfg):
    import yaml

    model_type = yaml.load(open(cfg), Loader=yaml.FullLoader)["model_type"]
    return {
        "regression": RegressionModel,
        "segmented_regression": SegmentedRegressionModel,
        "discretechoice": MNLDiscreteChoiceModel,
        "segmented_discretechoice": SegmentedMNLDiscreteChoiceModel,
    }[model_type]


def hedonic_simulate(cfg, tbl, nodes, out_fname):
    cfg = misc.config(cfg)
    df = to_frame([tbl, nodes], cfg)
    price_or_rent, _ = yaml_to_class(cfg).predict_from_cfg(df, cfg)

    if price_or_rent.replace([np.inf, -np.inf], np.nan).isnull().sum() > 0:
        print(
            "Hedonic output %d nas or inf (out of %d) in column %s"
            % (
                price_or_rent.replace([np.inf, -np.inf], np.nan).isnull().sum(),
                len(price_or_rent),
                out_fname,
            )
        )
    price_or_rent.loc[price_or_rent > 700] = 700
    price_or_rent.loc[price_or_rent < 1] = 1
    tbl.update_col_from_series(out_fname, price_or_rent, cast=True)


def lcm_simulate(
    cfg, choosers, buildings, nodes, out_fname, supply_fname, vacant_fname
):
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
    locations_df = to_frame([buildings, nodes], cfg, [supply_fname, vacant_fname])

    available_units = buildings[supply_fname]
    vacant_units = buildings[vacant_fname]

    print("There are %d total available units" % available_units.sum())
    print("    and %d total choosers" % len(choosers))
    print(
        "    but there are %d overfull buildings" % len(vacant_units[vacant_units < 0])
    )

    vacant_units = vacant_units[vacant_units > 0]
    units = locations_df.loc[
        np.repeat(vacant_units.index.values, vacant_units.values.astype("int"))
    ].reset_index()

    print("    for a total of %d temporarily empty units" % vacant_units.sum())
    print("    in %d buildings total in the region" % len(vacant_units))

    movers = choosers_df[choosers_df[out_fname] == -1]

    if len(movers) > vacant_units.sum():
        print("WARNING: Not enough locations for movers")
        print("    reducing locations to size of movers for performance gain")
        movers = movers.head(vacant_units.sum())

    new_units, _ = yaml_to_class(cfg).predict_from_cfg(movers, units, cfg)

    # new_units returns nans when there aren't enough units,
    # get rid of them and they'll stay as -1s
    new_units = new_units.dropna()

    # go from units back to buildings
    new_buildings = pd.Series(
        units.loc[new_units.values][out_fname].values, index=new_units.index
    )

    choosers.update_col_from_series(out_fname, new_buildings, cast=True)
    _print_number_unplaced(choosers, out_fname)

    vacant_units = buildings[vacant_fname]
    print("    and there are now %d empty units" % vacant_units.sum())
    print("    and %d overfull buildings" % len(vacant_units[vacant_units < 0]))


def simple_relocation(choosers, relocation_rate, fieldname):
    print("Total agents: %d" % len(choosers))
    _print_number_unplaced(choosers, fieldname)

    print("Assinging for relocation...")
    chooser_ids = np.random.choice(
        choosers.index, size=int(relocation_rate * len(choosers)), replace=False
    )
    choosers.update_col_from_series(fieldname, pd.Series(-1, index=chooser_ids))

    _print_number_unplaced(choosers, fieldname)


def simple_transition(tbl, rate, location_fname):
    transition = GrowthRateTransition(rate)
    df = tbl.to_frame(tbl.local_columns)

    print("%d agents before transition" % len(df.index))
    df, added, copied, removed = transition.transition(df, None)
    print("%d agents after transition" % len(df.index))

    df.loc[added, location_fname] = -1
    orca.add_table(tbl.name, df)


def _print_number_unplaced(df, fieldname):
    print("Total currently unplaced: %d" % df[fieldname].value_counts().get(-1, 0))


def random_choices(model, choosers, alternatives):
    """
    Simulate choices using random choice, weighted by probability
    but not capacity constrained.
    Parameters
    ----------
    model : SimulationChoiceModel
        Fitted model object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    alternatives : pandas.DataFrame
        DataFrame of alternatives.
    Returns
    -------
    choices : pandas.Series
        Mapping of chooser ID to alternative ID.
    """
    probabilities = model.calculate_probabilities(choosers, alternatives)
    choices = np.random.choice(
        probabilities.index, size=len(choosers), replace=True, p=probabilities.values
    )
    return pd.Series(choices, index=choosers.index)


def unit_choices(model, choosers, alternatives):
    """
    Simulate choices using unit choice.  Alternatives table is expanded
    to be of length alternatives.vacant_variables, then choices are simulated
    from among the universe of vacant units, respecting alternative capacity.
    Parameters
    ----------
    model : SimulationChoiceModel
        Fitted model object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    alternatives : pandas.DataFrame
        DataFrame of alternatives.
    Returns
    -------
    choices : pandas.Series
        Mapping of chooser ID to alternative ID.
    """
    supply_variable, vacant_variable = model.supply_variable, model.vacant_variable

    available_units = alternatives[supply_variable]
    vacant_units = alternatives[vacant_variable]
    vacant_units = vacant_units[
        vacant_units.index.values >= 0
    ]  ## must have positive index

    print("There are %d total available units" % available_units.sum())
    print("    and %d total choosers" % len(choosers))
    print(
        "    but there are %d overfull alternatives"
        % len(vacant_units[vacant_units < 0])
    )

    vacant_units = vacant_units[vacant_units > 0]

    indexes = np.repeat(vacant_units.index.values, vacant_units.values.astype("int"))
    isin = pd.Series(indexes).isin(alternatives.index)
    missing = len(isin[isin == False])
    indexes = indexes[isin.values]
    units = alternatives.loc[indexes].reset_index()

    print("    for a total of %d temporarily empty units" % vacant_units.sum())
    print("    in %d alternatives total in the region" % len(vacant_units))

    if missing > 0:
        print("WARNING: %d indexes aren't found in the locations df -" % missing)
        print("    this is usually because of a few records that don't join ")
        print("    correctly between the locations df and the aggregations tables")

    print("There are %d total movers for this LCM" % len(choosers))

    if len(choosers) > vacant_units.sum():
        print("WARNING: Not enough locations for movers")
        print("    reducing locations to size of movers for performance gain")
        choosers = choosers.head(vacant_units.sum())

    choices = model.predict(choosers, units, debug=True)

    def identify_duplicate_choices(choices):
        choice_counts = choices.value_counts()
        return choice_counts[choice_counts > 1].index.values

    if model.choice_mode == "individual":
        print("Choice mode is individual, so utilizing lottery choices.")

        chosen_multiple_times = identify_duplicate_choices(choices)

        while len(chosen_multiple_times) > 0:
            duplicate_choices = choices[choices.isin(chosen_multiple_times)]

            # Identify the choosers who keep their choice, and those who must
            # choose again.
            keep_choice = duplicate_choices.drop_duplicates()
            rechoose = duplicate_choices[
                ~duplicate_choices.index.isin(keep_choice.index)
            ]

            # Subset choices, units, and choosers to account for occupied
            # units and choosers who need to choose again.
            choices = choices.drop(rechoose.index)
            units_remaining = units.drop(choices.values)
            choosers = choosers.drop(choices.index)

            # Agents choose again.
            next_choices = model.predict(choosers, units_remaining)
            choices = pd.concat([choices, next_choices])
            chosen_multiple_times = identify_duplicate_choices(choices)

    return pd.Series(
        units.loc[choices.values][model.choice_column].values, index=choices.index
    )


class SimulationChoiceModel(MNLDiscreteChoiceModel):
    """
    A discrete choice model with parameters needed for simulation.
    Initialize with MNLDiscreteChoiceModel's init parameters or with from_yaml, 
    then add simulation parameters with set_simulation_params().

    """

    def set_simulation_params(
        self,
        name,
        supply_variable,
        vacant_variable,
        choosers,
        alternatives,
        summary_alts_xref=None,
    ):
        """
        Add simulation parameters as additional attributes.
        Parameters
        ----------
        name : str
            Name of the model.
        supply_variable : str
            The name of the column in the alternatives table indicating number
            of available spaces, vacant or not, that can be occupied by
            choosers.
        vacant_variable : str
            The name of the column in the alternatives table indicating number
            of vacant spaces that can be occupied by choosers.
        choosers : str
            Name of the choosers table.
        alternatives : str
            Name of the alternatives table.
        summary_alts_xref : dict or pd.Series, optional
            Mapping of alternative index to summary alternative id.  For use
            in evaluating a model with many alternatives.
        Returns
        -------
        None
        """
        self.name = name
        self.supply_variable = supply_variable
        self.vacant_variable = vacant_variable
        self.choosers = choosers
        self.alternatives = alternatives
        self.summary_alts_xref = summary_alts_xref

    def simulate(self, choice_function=None, save_probabilities=False, **kwargs):
        """
        Computing choices, with arbitrary function for handling simulation strategy. 
        Parameters
        ----------
        choice_function : function
            Function defining how to simulate choices based on fitted model.
            Function must accept the following 3 arguments:  model object, choosers
            DataFrame, and alternatives DataFrame.  Additional optional keyword
            args can be utilized by function if needed (kwargs).
        save_probabilities : bool
            If true, will save the calculated probabilities underlying the simulation 
            as an orca injectable with name 'probabilities_modelname_itervar'.
        Returns
        -------
        choices : pandas.Series
            Mapping of chooser ID to alternative ID. Some choosers
            will map to a nan value when there are not enough alternatives
            for all the choosers.
        """
        choosers, alternatives = self.calculate_model_variables()

        # By convention, choosers are denoted by a -1 value in the choice column
        choosers = choosers[choosers[self.choice_column] == -1]
        print("%s agents are making a choice." % len(choosers))

        if choice_function:
            choices = choice_function(self, choosers, alternatives, **kwargs)
        else:
            choices = self.predict(choosers, alternatives, debug=True)

        if save_probabilities:
            if not self.sim_pdf:
                probabilities = self.calculate_probabilities(
                    self, choosers, alternatives
                )
            else:
                probabilities = self.sim_pdf.reset_index().set_index("alternative_id")[
                    0
                ]
            orca.add_injectable(
                "probabilities_%s_%s" % (self.name, orca.get_injectable("iter_var")),
                probabilities,
            )

        return choices

    def calculate_probabilities(self, choosers, alternatives):
        """
        Calculate model probabilities.
        Parameters
        ----------
        choosers : pandas.DataFrame
            DataFrame of choosers.
        alternatives : pandas.DataFrame
            DataFrame of alternatives.
        Returns
        -------
        probabilities : pandas.Series
            Mapping of alternative ID to probabilities.
        """
        probabilities = self.probabilities(choosers, alternatives)
        probabilities = probabilities.reset_index().set_index("alternative_id")[
            0
        ]  # remove chooser_id col from idx
        return probabilities

    def calculate_model_variables(self):
        """
        Calculate variables needed to simulate the model, and returns DataFrames 
        of simulation-ready tables with needed variables.
        Returns
        -------
        choosers : pandas.DataFrame
            DataFrame of choosers.
        alternatives : pandas.DataFrame
            DataFrame of alternatives.
        """
        columns_used = self.columns_used() + [self.choice_column]
        choosers = orca.get_table(self.choosers).to_frame(columns_used)

        supply_column_names = [
            col
            for col in [self.supply_variable, self.vacant_variable]
            if col is not None
        ]
        alternatives = orca.get_table(self.alternatives).to_frame(
            columns_used + supply_column_names
        )
        return choosers, alternatives

    def score(
        self,
        scoring_function=accuracy_score,
        choosers=None,
        alternatives=None,
        aggregate=False,
        apply_filter=True,
    ):
        """
        Calculate score for model.  Defaults to accuracy score, but other
        scoring functions can be provided.  Computed on all choosers/
        alternatives by default, but can also be computed on user-supplied
        test datasets.  If model has a summary_alts_xref, then score
        calculated after mapping to summary ids.
        Parameters
        ----------
        scoring_function : function, default sklearn.metrics.accuracy_score
            Function defining how to score model predictions. Function must
            accept the following 2 arguments:  pd.Series of observed choices,
            pd.Series of predicted choices.
        choosers : pandas.DataFrame, optional
            DataFrame of choosers.
        alternatives : pandas.DataFrame, optional
            DataFrame of alternatives.
        aggregate : bool
            Whether to calculate score based on total count of choosers that
            made each choice, rather than based on disaggregate choices.
        apply_filter : bool
            Whether to apply the model's choosers_predict_filters prior to
            calculating score.  If supplying own test dataset, and do not want
            it further manipulated, then set to False.
        Returns
        -------
        score : float
            The model's score (accuracy score by default).
        """
        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables()

        if apply_filter:
            choosers = choosers.query(self.choosers_predict_filters)

        choosers = choosers[
            (~choosers[self.choice_column].isnull())
            | (choosers[self.choice_column] != -1)
        ]
        observed_choices = choosers[self.choice_column].astype("int")
        predicted_choices = random_choices(self, choosers, alternatives)

        if self.summary_alts_xref is not None:
            observed_choices = observed_choices.map(self.summary_alts_xref)
            predicted_choices = predicted_choices.map(self.summary_alts_xref)

        if aggregate:
            observed_choices = observed_choices.value_counts()
            predicted_choices = predicted_choices.value_counts()
        try:
            return scoring_function(observed_choices, predicted_choices)
        except:
            import pdb

            pdb.set_trace()


def apply_filter_query(df, filters=None):
    """
    Use the DataFrame.query method to filter a table down to the
    desired rows.

    Parameters
    ----------
    df : pandas.DataFrame
    filters : list of str or str, optional
        List of filters to apply. Will be joined together with
        ' and ' and passed to DataFrame.query. A string will be passed
        straight to DataFrame.query.
        If not supplied no filtering will be done.

    Returns
    -------
    filtered_df : pandas.DataFrame

    """
    if filters:
        if isinstance(filters, str):
            query = filters
        else:
            query = " and ".join(filters)
        return df.query(query)
    else:
        return df


def _filterize(name, value):
    """
    Turn a `name` and `value` into a string expression compatible
    the ``DataFrame.query`` method.

    Parameters
    ----------
    name : str
        Should be the name of a column in the table to which the
        filter will be applied.

        A suffix of '_max' will result in a "less than" filter,
        a suffix of '_min' will result in a "greater than or equal to" filter,
        and no recognized suffix will result in an "equal to" filter.
    value : any
        Value side of filter for comparison to column values.

    Returns
    -------
    filter_exp : str

    """
    if name.endswith("_min"):
        name = name[:-4]
        comp = ">="
    elif name.endswith("_max"):
        name = name[:-4]
        comp = "<"
    else:
        comp = "=="

    result = "{} {} {!r}".format(name, comp, value)
    return result


def filter_table(table, filter_series, ignore=None):
    """
    Filter a table based on a set of restrictions given in
    Series of column name / filter parameter pairs. The column
    names can have suffixes `_min` and `_max` to indicate
    "less than" and "greater than" constraints.

    Parameters
    ----------
    table : pandas.DataFrame
        Table to filter.
    filter_series : pandas.Series
        Series of column name / value pairs of filter constraints.
        Columns that ends with '_max' will be used to create
        a "less than" filters, columns that end with '_min' will be
        used to create "greater than or equal to" filters.
        A column with no suffix will be used to make an 'equal to' filter.
    ignore : sequence of str, optional
        List of column names that should not be used for filtering.

    Returns
    -------
    filtered : pandas.DataFrame

    """
    ignore = ignore if ignore else set()

    filters = [
        _filterize(name, val)
        for name, val in filter_series.items()
        if not (name in ignore or (isinstance(val, numbers.Number) and np.isnan(val)))
    ]

    return apply_filter_query(table, filters)


def run_log(log_txt, file_path=None):
    """
    append run info to log file
    file_path: log file folder, same as indicator folder
    """
    if file_path is not None:
        data_out_dir = file_path
    elif "data_out_dir" in orca.list_injectables():
        data_out_dir = orca.get_injectable("data_out_dir")
    else:
        data_out_dir = "./"

    if not (os.path.exists(data_out_dir)):
        os.makedirs(data_out_dir)

    with open(os.path.join(data_out_dir, "run_log.txt"), "a") as logf:
        logf.write(log_txt + "\n")


def debug_log(file_path=None):
    """
    write all logging DEBUG info to file
    """
    if file_path is not None:
        data_out_dir = file_path
    elif "data_out_dir" in orca.list_injectables():
        data_out_dir = orca.get_injectable("data_out_dir")
    else:
        data_out_dir = "./"

    if not (os.path.exists(data_out_dir)):
        os.makedirs(data_out_dir)

    logging.basicConfig(
        filename=os.path.join(data_out_dir, "run_debug.txt"),
        filemode="w",
        format="%(message)s",
        level=logging.DEBUG,
        force=True,
    )


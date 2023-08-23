

import os
import copy
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

import orca
from urbansim.utils import misc
from urbansim.models import dcm
from urbansim.models import util
from urbansim.urbanchoice import interaction
from urbansim.models import MNLDiscreteChoiceModel
from urbansim_templates.models import LargeMultinomialLogitStep
from urbansim.models.util import (apply_filter_query, columns_in_filters, 
        columns_in_formula)



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
        probabilities.index, size=len(choosers),
        replace=True, p=probabilities.values)
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
    supply_variable, vacant_variable = (model.supply_variable,
                                        model.vacant_variable)

    available_units = alternatives[supply_variable]
    vacant_units = alternatives[vacant_variable]
    # must have positive index
    vacant_units = vacant_units[vacant_units.index.values >= 0]

    print("There are {} total available units"
          .format(available_units.sum()),
          "    and {} total choosers"
          .format(len(choosers)),
          "    but there are {} overfull alternatives"
          .format(len(vacant_units[vacant_units < 0])))

    vacant_units = vacant_units[vacant_units > 0]

    indexes = np.repeat(vacant_units.index.values,
                        vacant_units.values.astype('int'))
    isin = pd.Series(indexes).isin(alternatives.index)
    missing = len(isin[isin == False])  # noqa
    indexes = indexes[isin.values]
    units = alternatives.loc[indexes].reset_index()

    print("    for a total of {} temporarily empty units"
          .format(vacant_units.sum()),
          "    in {} alternatives total in the region"
          .format(len(vacant_units)))

    if missing > 0:
        print(
            "WARNING: {} indexes aren't found in the locations df -"
            .format(missing),
            "    this is usually because of a few records that don't join ",
            "    correctly between the locations df and the aggregations",
            "tables")

    print("There are {} total movers for this LCM".format(len(choosers)))

    if len(choosers) > vacant_units.sum():
        print("WARNING: Not enough locations for movers",
              "reducing locations to size of movers for performance gain")
        choosers = choosers.head(int(vacant_units.sum()))

    choices = model.predict(choosers, units, debug=True)

    def identify_duplicate_choices(choices):
        choice_counts = choices.value_counts()
        return choice_counts[choice_counts > 1].index.values

    if model.choice_mode == 'individual':
        print('Choice mode is individual, so utilizing lottery choices.')

        chosen_multiple_times = identify_duplicate_choices(choices)

        while len(chosen_multiple_times) > 0:
            duplicate_choices = choices[choices.isin(chosen_multiple_times)]

            # Identify the choosers who keep their choice, and those who must
            # choose again.
            keep_choice = duplicate_choices.drop_duplicates()
            rechoose = duplicate_choices[~duplicate_choices.index.isin(
                                                           keep_choice.index)]

            # Subset choices, units, and choosers to account for occupied
            # units and choosers who need to choose again.
            choices = choices.drop(rechoose.index)
            units_remaining = units.drop(choices.values)
            choosers = choosers.drop(choices.index, errors='ignore')

            # Agents choose again.
            next_choices = model.predict(choosers, units_remaining)
            choices = pd.concat([choices, next_choices])
            chosen_multiple_times = identify_duplicate_choices(choices)

    return pd.Series(units.loc[choices.values][model.choice_column].values,
                     index=choices.index)


def register_config_injectable_from_yaml(injectable_name, yaml_file):
    """
    Create orca function for YAML-based config injectables.
    """
    @orca.injectable(injectable_name, cache=True)
    def func():
        with open(os.path.join(misc.configs_dir(), yaml_file)) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return config
    return func


def register_choice_model_step(model_name, agents_name):

    @orca.step(model_name)
    def choice_model_simulate(location_choice_models):
        model = location_choice_models[model_name]
        if 'hlcm' in model_name:
            alts_pre_filter = chooser_pre_filter = "(large_area_id==%s)" % (model_name.split('_')[1])
            # filter for picking hh with no building_id assigned
            chooser_filter = "(building_id==-1)"
            alt_filter = "(residential_units>0) & (mcd_model_quota>0) & (hu_filter==0) & (sp_filter>=0)"
        elif 'elcm' in model_name:
            chooser_pre_filter = "(slid==%s) & (home_based_status==0)" % (model_name.split('_')[1])
            alts_pre_filter = "(large_area_id==%s)" % (int(model_name.split('_')[1]) % 1000)
            # filter for picking jobs with not building_id assigned
            chooser_filter = "(building_id==-1)"
            alt_filter = "(non_residential_sqft>0)&(sp_filter>=0)"
            
        # initialize simulation choosers and alts table
        formula_cols = columns_in_formula(model.model_expression)
        choosers_filter_cols = columns_in_filters(chooser_filter) + columns_in_filters(chooser_pre_filter)
        alts_filter_cols = columns_in_filters(alt_filter) + columns_in_filters(alts_pre_filter)
        # choosers
        choosers = orca.get_table(model.choosers)
        formula_chooser_col = [col for col in formula_cols if col in choosers.columns]
        choosers_df = choosers.to_frame(formula_chooser_col+choosers_filter_cols)
        # query using chooser_pre_filter to match whats used in estimation
        choosers_idx = choosers_df.query(chooser_pre_filter).index
        # std choosers columns
        chooser_col_df = choosers_df.loc[choosers_idx, formula_chooser_col]
        choosers_df.loc[choosers_idx, formula_chooser_col] = (
            chooser_col_df-chooser_col_df.mean())/chooser_col_df.std()
        # filter using chooser_filter
        final_choosers_df = choosers_df.loc[choosers_idx].query(chooser_filter)

        # alternatives
        alts = orca.get_table(model.alternatives)
        formula_alts_col = [col for col in formula_cols if col in alts.columns]
        alts_df = alts.to_frame(formula_alts_col+alts_filter_cols+[model.alt_capacity])
        # query using alts_pre_filter to match whats used in estimation
        alts_idx = alts_df.query(alts_pre_filter).index
        # std alts columns
        alts_col_df = alts_df.loc[alts_idx, formula_alts_col]
        # std could introduce NaN, fill them with 0 after that
        alts_df.loc[alts_idx, formula_alts_col] = ((
            alts_col_df-alts_col_df.mean())/alts_col_df.std()).fillna(0)

        # filter using alt_filter
        final_alts_df = alts_df.loc[alts_idx].query(alt_filter)

        orca.add_table('choosers', final_choosers_df)
        orca.add_table('alternatives', final_alts_df)
        
        model.out_choosers = 'choosers'
        model.out_chooser_filters = None # already filtered
        model.out_alternatives = 'alternatives'
        model.out_alt_filters = None # already filtered

        model.run(chooser_batch_size=1000)

        # if not choices, return
        if not type(model.choices) == pd.Series:
            print('There are 0 unplaced agents.')
            return

        print('There are {} unplaced agents.'
              .format(model.choices.isnull().sum()))

        orca.get_table(agents_name).update_col_from_series(
            model.choice_column, model.choices, cast=True)

    return choice_model_simulate


class SimulationChoiceModel(MNLDiscreteChoiceModel):
    """
    A discrete choice model with parameters needed for simulation.
    Initialize with MNLDiscreteChoiceModel's init parameters or with from_yaml,
    then add simulation parameters with set_simulation_params().

    """
    def set_simulation_params(self, name, supply_variable, vacant_variable,
                              choosers, alternatives, choice_column=None,
                              summary_alts_xref=None, merge_tables=None,
                              agent_units=None):
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
        merge_tables : list of str, optional
            List of additional tables to be broadcast onto the alternatives
            table.
        agent_units : str, optional
            Name of the column in the choosers table that designates how
            much supply is occupied by each chooser.
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
        self.merge_tables = merge_tables
        self.agent_units = agent_units
        self.choice_column = choice_column if choice_column is not None \
            else self.choice_column

    def simulate(self, choice_function=None, save_probabilities=False,
                 **kwargs):
        """
        Computing choices, with arbitrary function for handling simulation
        strategy.
        Parameters
        ----------
        choice_function : function
            Function defining how to simulate choices based on fitted model.
            Function must accept the following 3 arguments:  model object,
            choosers DataFrame, and alternatives DataFrame. Additional optional
            keyword args can be utilized by function if needed (kwargs).
        save_probabilities : bool
            If true, will save the calculated probabilities underlying the
            simulation as an orca injectable with name
            'probabilities_modelname_itervar'.
        Returns
        -------
        choices : pandas.Series
            Mapping of chooser ID to alternative ID. Some choosers
            will map to a nan value when there are not enough alternatives
            for all the choosers.
        """
        choosers, alternatives = self.calculate_model_variables()

        choosers, alternatives = self.apply_predict_filters(
                                 choosers, alternatives)

        # By convention, choosers are denoted by a -1 value
        # in the choice column
        choosers = choosers[choosers[self.choice_column] == -1]
        print("{} agents are making a choice.".format(len(choosers)))

        if choice_function:
            choices = choice_function(self, choosers, alternatives, **kwargs)
        else:
            choices = self.predict(choosers, alternatives, debug=True)

        if save_probabilities:
            if not self.sim_pdf:
                probabilities = self.calculate_probabilities(choosers,
                                                             alternatives)
            else:
                probabilities = self.sim_pdf.reset_index().set_index(
                    'alternative_id')[0]
            orca.add_injectable('probabilities_{}_{}'.format(
                self.name, orca.get_injectable('iter_var')),
                probabilities)

        return choices

    def fit_model(self):
        """
        Estimate model based on existing parameters
        Returns
        -------
        None
        """
        choosers, alternatives = self.calculate_model_variables()
        self.fit(choosers, alternatives, choosers[self.choice_column])
        return self.log_likelihoods, self.fit_parameters

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
        probabilities = probabilities.reset_index().set_index(
            'alternative_id')[0]  # remove chooser_id col from idx
        return probabilities

    def calculate_model_variables(self):
        """
        Calculate variables needed to simulate the model, and returns
        DataFrames of simulation-ready tables with needed variables.
        Returns
        -------
        choosers : pandas.DataFrame
            DataFrame of choosers.
        alternatives : pandas.DataFrame
            DataFrame of alternatives.
        """
        columns_used = self.columns_used() + [self.choice_column]
        columns_used = columns_used + [self.agent_units] if self.agent_units else columns_used
        choosers = orca.get_table(self.choosers).to_frame(columns_used)

        supply_column_names = [col for col in
                               [self.supply_variable, self.vacant_variable]
                               if col is not None]

        columns_used.extend(supply_column_names)

        if self.merge_tables:
            mt = copy.deepcopy(self.merge_tables)
            mt.append(self.alternatives)
            all_cols = []
            for table in mt:
                all_cols.extend(orca.get_table(table).columns)
            all_cols = [col for col in all_cols if col in columns_used]
            alternatives = orca.merge_tables(target=self.alternatives,
                                             tables=mt, columns=all_cols)
        else:
            alternatives = orca.get_table(self.alternatives).to_frame(
                columns_used + supply_column_names)
        return choosers, alternatives

    def score(self, scoring_function=accuracy_score, choosers=None,
              alternatives=None, aggregate=False, apply_filter=True,
              choice_function=random_choices):
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
        choice_function : function, option
            Function defining how to simulate choices.
        Returns
        -------
        score : float
            The model's score (accuracy score by default).
        """
        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables()

        if apply_filter:
            if self.choosers_predict_filters:
                choosers = choosers.query(self.choosers_predict_filters)
            if self.choosers_fit_filters:
                choosers = choosers.query(self.choosers_fit_filters)

        observed_choices = choosers[self.choice_column]
        predicted_choices = choice_function(self, choosers, alternatives)

        if self.summary_alts_xref is not None:
            observed_choices = observed_choices.map(self.summary_alts_xref)
            predicted_choices = predicted_choices.map(self.summary_alts_xref)

        if aggregate:
            observed_choices = observed_choices.value_counts()
            predicted_choices = predicted_choices.value_counts()

            combined_index = list(set(list(predicted_choices.index) +
                                      list(observed_choices.index)))
            predicted_choices = predicted_choices.reindex(combined_index).fillna(0)
            observed_choices = observed_choices.reindex(combined_index).fillna(0)

        return scoring_function(observed_choices, predicted_choices)

    def summed_probabilities(self, choosers=None, alternatives=None):
        """
        Sum probabilities to the summary geography level.
        """
        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables()

        if self.choosers_fit_filters:
            choosers = choosers.query(self.choosers_fit_filters)

        if self.choosers_predict_filters:
            choosers = choosers.query(self.choosers_predict_filters)

        choosers['summary_id'] = choosers[self.choice_column]
        choosers.summary_id = choosers.summary_id.map(self.summary_alts_xref)
        probs = self.calculate_probabilities(choosers, alternatives)
        probs = probs.reset_index().rename(columns={0: 'proba'})
        probs['summary_id'] = probs.alternative_id.map(self.summary_alts_xref)
        return probs.groupby('summary_id').proba.sum()

    def observed_distribution(self, choosers=None):
        """
        Calculate observed distribution across alternatives at the summary
        geography level.
        """
        if choosers is None:
            choosers, alternatives = self.calculate_model_variables()

        if self.choosers_fit_filters:
            choosers = choosers.query(self.choosers_fit_filters)

        if self.choosers_predict_filters:
            choosers = choosers.query(self.choosers_predict_filters)

        if 'summary_id' not in choosers.columns:
            summ_id = choosers[self.choice_column].map(self.summary_alts_xref)
            choosers['summary_id'] = summ_id

        observed_distrib = choosers.groupby('summary_id').size()
        return observed_distrib / observed_distrib.sum()

    def summed_probability_score(self, scoring_function=r2_score,
                                 choosers=None, alternatives=None,
                                 validation_data=None):
        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables()

        if self.choosers_fit_filters:
            choosers = choosers.query(self.choosers_fit_filters)

        if self.choosers_predict_filters:
            choosers = choosers.query(self.choosers_predict_filters)

        summed_probas = self.summed_probabilities(choosers, alternatives)

        if validation_data is None:
            validation_data = self.observed_distribution(choosers)

        combined_index = list(set(list(summed_probas.index) +
                                  list(validation_data.index)))
        summed_probas = summed_probas.reindex(combined_index).fillna(0)
        validation_data = validation_data.reindex(combined_index).fillna(0)

        print(summed_probas.corr(validation_data))
        score = scoring_function(validation_data, summed_probas)
        print(score)

        residuals = summed_probas - validation_data
        return score, residuals


def get_model_category_configs():
    """
    Returns dictionary where key is model category name and value is dictionary
    of model category attributes, including individual model config filename(s)
    """
    with open(os.path.join(misc.configs_dir(), 'yaml_configs_2050.yaml')) as f:
        yaml_configs = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(misc.configs_dir(), 'model_structure.yaml')) as f:
        model_category_configs = yaml.load(f, Loader=yaml.FullLoader)['models']

    for model_category, category_attributes in list(model_category_configs.items()):
        category_attributes['config_filenames'] = yaml_configs[model_category]

    return model_category_configs


def create_lcm_from_config(config_filename, model_attributes):
    """
    For a given model config filename and dictionary of model category
    attributes, instantiate a LargeMultinomialLogitStep object.

    config_filename: model name
    model_attributes: model_structure.yaml
    """
    with open(misc.config(config_filename), "r") as f:
        config_obj = yaml.load(f, Loader=yaml.FullLoader)

    model = LargeMultinomialLogitStep.from_dict(config_obj['saved_object'])
    model.choosers = model_attributes['agents_name']
    model.alternatives = model_attributes['alternatives_name']
    model.choice_column = model_attributes['alternatives_id_name']
    # is it alt_capacity in largeMNL equals vacant_variable in 2045?
    model.alt_capacity = model_attributes['vacant_variable']
    return model

def get_hlcm_valid_vars(data_path: str) -> tuple[list[str], list[str]]:
    """
    Extract valid household and building variable names from a YAML configuration file.

    Parameters:
    data_path (str): Path to the directory containing the variable validation YAML files.

    Returns:
    tuple: A tuple containing two lists of valid variable names: valid household variable names and valid building variable names.
    """
    var_validation_list = [
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if ("variable_validation" in f) and (f[-5:] == ".yaml")
    ]
    var_validation_last = max(var_validation_list, key=os.path.getctime)
    
    with open(var_validation_last, "r") as f:
        vars_config = yaml.load(f, Loader=yaml.FullLoader)
    
    valid_b_vars = vars_config["buildings"]["valid variables"]
    valid_hh_vars = vars_config["households"]["valid variables"]
    return valid_hh_vars, valid_b_vars

def get_elcm_valid_vars(data_path: str) -> tuple[list[str], list[str]]:
    """
    Extract valid household and building variable names from a YAML configuration file.

    Parameters:
    data_path (str): Path to the directory containing the variable validation YAML files.

    Returns:
    tuple: A tuple containing two lists of valid variable names: valid household variable names and valid building variable names.
    """
    var_validation_list = [
        (data_path + "/" + f)
        for f in os.listdir(data_path)
        if ("variable_validation" in f) & (f[-5:] == ".yaml")
    ]
    var_validation_last = max(var_validation_list, key=os.path.getctime)
    with open(var_validation_last, "r") as f:
        vars_config = yaml.load(f, Loader=yaml.FullLoader)
    valid_job_vars = vars_config["jobs"]["valid variables"]
    valid_b_vars = vars_config["buildings"]["valid variables"]
    return valid_job_vars, valid_b_vars

def load_hlcm_df(households: orca.DataFrameWrapper, buildings: orca.DataFrameWrapper, hh_var: list[str], b_var: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and return household and building DataFrames.

    Parameters:
    households (orca.DataFrameWrapper): Orca DataFrameWrapper for households.
    buildings (orca.DataFrameWrapper): Orca DataFrameWrapper for buildings.
    hh_var (list[str]): Names of the household variables to load.
    b_var (list[str]): Names of the building variables to load.

    Returns:
    tuple: A tuple containing two DataFrames: the household DataFrame and the building DataFrame.

    Example:
    >>> household_df, building_df = load_hlcm_df(households, buildings, ['persons'], ['parcel_id'])
    """
    hh = households.to_frame(hh_var)
    b = buildings.to_frame(b_var)
    return hh, b

def load_elcm_df(jobs: orca.DataFrameWrapper, buildings: orca.DataFrameWrapper, job_var: list[str], b_var: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and return job and building DataFrames.

    Parameters:
    jobs (orca.DataFrameWrapper): Jobs data DataFrame.
    buildings (orca.DataFrameWrapper): Building data DataFrame.
    job_var (list[str]): Names of the job variables to load.
    b_var (list[str]): Names of the building variables to load.

    Returns:
    tuple: A tuple containing two DataFrames: the job DataFrame and the building DataFrame.

    Example:
    >>> job_df, building_df = load_elcm_df(['employment'], ['parcel_id'])
    """
    job_df = jobs.to_frame(job_var)
    b_df = buildings.to_frame(b_var)
    return job_df, b_df

def columns_in_vars(vars: list[str], valid_agent_vars: list[str], valid_b_vars: list[str]) -> tuple[list[str], list[str]]:
    """
    Categorize variables into agents and building columns.

    This function takes a list of variable names and categorizes them into
    agents columns and building columns based on the presence of a colon
    separator or matching valid variable names.

    Parameters:
    vars (list[str]): List of variable names to categorize.
    valid_agent_vars (list[str]): List of valid agents variable names
    valid_b_vars (list[str]): List of valid building variable names

    Returns:
    tuple: A tuple containing two lists of strings: agents column names and building column names.
    """
    agent_columns, b_columns = [], []
    for varname in vars:
        if ':' in varname:
            agent_col, b_col = map(str.strip, varname.split(':'))
            if agent_col in valid_agent_vars:
                agent_columns.append(agent_col)
            if b_col in valid_b_vars:
                b_columns.append(b_col)
        elif varname in valid_agent_vars:
            agent_columns.append(varname)
        elif varname in valid_b_vars:
            b_columns.append(varname)
        else:
            print(varname, " not found in both agents and buildings table")
    return agent_columns, b_columns

def get_interaction_vars(df: pd.DataFrame, varname: str) -> np.ndarray:
    """
    Get interaction variables from variable name.

    This function calculates interaction variables based on the provided variable name
    within the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the variables.
        varname (str): The name of the interaction variable.

    Returns:
        np.ndarray: A NumPy array containing the calculated interaction variables.

    Example:
    >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    >>> df = pd.DataFrame(data)
    >>> interaction_array = get_interaction_vars(df, 'A:B')
    """
    if ":" in varname:
        var1, var2 = map(str.strip, varname.split(":"))
        return (df[var1] * df[var2]).values.reshape(-1, 1)
    else:
        return df[varname].values.reshape(-1, 1)

def load_hlcm_dataset(valid_hh_vars, valid_b_vars, var_pool_table_path, hh_filter_columns, b_filter_columns, use_cache=False):
    """
    Load and preprocess dataset variables

    This function loads and preprocesses the dataset variables needed for estimation. It extracts the set of valid variables
    from the variable pool table, loads the necessary variables from the 'buildings' and 'households' Orca tables,
    and caches the resulting household and building DataFrames for later use.

    Parameters:
    valid_hh_vars (list): A list of valid household variable names.
    valid_b_vars (list): A list of valid building variable names.

    Returns:
    tuple: A tuple containing the following elements:
        - hh_region (pd.DataFrame): A DataFrame containing loaded household data.
        - b_region (pd.DataFrame): A DataFrame containing loaded building data.
        - vars_to_use (np.ndarray): An array of variable names used for modeling.
    """
    # Load the variable pool table and extract valid variable names
    used_vars = pd.read_excel(var_pool_table_path, sheet_name=2)
    v1 = used_vars[~used_vars["new variables 1"].isna()]["new variables 1"].unique()
    v2 = used_vars[~used_vars["new variables 2"].isna()]["new variables 2"].unique()
    vars_to_use = np.array(list(set(v1.tolist()).union(v2.tolist())))

    # Choose whether to reload data or use cached data
    if not use_cache:
        # from notebooks.models_test import *
        import models
        buildings = orca.get_table("buildings")
        households = orca.get_table("households")

        # set year to 2020 and run build network and neigh vars
        orca.add_injectable('year', 2020)
        orca.run(["build_networks_2050"])
        orca.run(["neighborhood_vars"])

        # set year to 2050 and run mcd_hu_sampling
        orca.add_injectable('year', 2050)
        orca.run(["mcd_hu_sampling"])

        # Get valid variables for modeling and load corresponding data
        hh_columns, b_columns = columns_in_vars(vars_to_use, valid_hh_vars, valid_b_vars)
        hh_var = hh_columns + hh_filter_columns
        b_var = b_columns + b_filter_columns
        hh_region, b_region = load_hlcm_df(households, buildings, hh_var, b_var)

        # Cache the loaded DataFrames as CSV files
        hh_region.to_csv('data/hh.csv')
        b_region.to_csv('data/b_hlcm.csv')
    else:
        hh_region, b_region = load_cache_hh_b('data/hh.csv', 'data/b_hlcm.csv')
    return hh_region, b_region, vars_to_use

def load_elcm_dataset(valid_job_vars, valid_b_vars, var_pool_table_path, job_filter_columns, b_filter_columns, use_cache=False):
    """
    Load and preprocess job and building datasets for ELCM estimation.

    This function loads the job and building datasets, extracts valid variable names, and preprocesses the data
    for estimation using the ELCM (Employment Location Choice Model).

    Parameters:
    valid_job_vars (list[str]): Valid job variables for modeling.
    valid_b_vars (list[str]): Valid building variables for modeling.
    var_pool_table_path (str): Path to the variable pool table Excel file.
    job_filter_columns (list[str]): Job filter columns to exclude from the loaded data.
    b_filter_columns (list[str]): Building filter columns to exclude from the loaded data.
    use_cache (Boolean): Use cache(True) or reload(False)

    Returns:
    tuple: A tuple containing job_region DataFrame, building_region DataFrame, and vars_to_use array.

    Example:
    >>> job_region, building_region, vars_to_use = load_elcm_dataset(valid_job_vars, valid_b_vars,
    ...                                                              var_pool_table_path, job_filter_columns,
    ...                                                              b_filter_columns)
    """
    # Load the variable pool table and extract valid variable names
    used_vars = pd.read_excel(var_pool_table_path, sheet_name=1)
    v1 = used_vars[~used_vars["variables 1"].isna()]["variables 1"].unique()
    v2 = used_vars[~used_vars["Variables 2"].isna()]["Variables 2"].unique()
    vars_to_use = np.array(list(set(v1.tolist()).union(v2.tolist())))

    # Choose whether to reload data or use cached data
    if not use_cache:
        # from notebooks.models_test import *
        import models
        buildings = orca.get_table("buildings")
        jobs = orca.get_table("jobs")

        # set year to 2020 and run build network and neigh vars
        orca.add_injectable('year', 2020)
        orca.run(["build_networks_2050"])
        orca.run(["neighborhood_vars"])

        # set year to 2050 and run mcd_hu_sampling
        orca.add_injectable('year', 2050)
        orca.run(["mcd_hu_sampling"])

        # Get valid variables for modeling and load corresponding data
        job_columns, b_columns = columns_in_vars(vars_to_use, valid_job_vars, valid_b_vars)

        job_var = job_columns + job_filter_columns
        b_var = b_columns + b_filter_columns
        job_region, b_region = load_elcm_df(jobs, buildings, job_var, b_var)

        # Cache the loaded DataFrames as CSV files
        job_region.to_csv('data/jobs.csv')
        b_region.to_csv('data/b_elcm.csv')
    else:
        job_region, b_region = load_cache_hh_b('data/jobs.csv', 'data/b_elcm.csv')
    return job_region, b_region, vars_to_use

def load_cache_hh_b(hh_csv_path: str, b_csv_path: str):
    """
    Load household and building data from CSV files and register them as tables.

    Parameters:
    hh_csv_path (str): Path to the household CSV file.
    b_csv_path (str): Path to the building CSV file.
    """
    try:
        hh_region = pd.read_csv(hh_csv_path, index_col=0)
        b_region = pd.read_csv(b_csv_path, index_col=0)
    except FileNotFoundError:
        print("CSV file not found. Please provide correct file paths.")
        return 
    
    orca.add_table('households', hh_region)
    orca.add_table('buildings', b_region)
    return hh_region, b_region

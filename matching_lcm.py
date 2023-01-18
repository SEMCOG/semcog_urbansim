from __future__ import print_function

import pandas as pd
import numpy as np
import numba
import orca
from choicemodels.tools import MergedChoiceTable
from urbansim_templates import modelmanager
from urbansim_templates.models import TemplateStep
from urbansim_templates.utils import get_data, update_column, to_list, version_greater_or_equal
from urbansim.models.util import columns_in_formula, apply_filter_query

def check_choicemodels_version():
    try:
        import choicemodels
        assert version_greater_or_equal(choicemodels.__version__, '0.2.dev4')
    except:
        raise ImportError("LargeMultinomialLogitStep requires choicemodels 0.2.dev4 or "
                          "later. For installation instructions, see "
                          "https://github.com/udst/choicemodels.")

@modelmanager.template
class MatchingLocationChoiceModel(TemplateStep):
    """
	MatchingLocationChoiceModel
	Using matching algorithm to match agents to alts given distance between them

    Parameters
    ----------
    choosers : str or list of str, optional
        Name(s) of Orca tables to draw choice scenario data from. The first table is the
        primary one. Any additional tables need to have merge relationships ("broadcasts")
        specified so that they can be merged unambiguously onto the first table. The index
        of the primary table should be a unique ID. In this template, the 'choosers' and
        'alternatives' parameters replace the 'tables' parameter. Both are required for
        fitting a model, but do not have to be provided when the object is created.
        Reserved column names: 'chosen'.

    alternatives : str or list of str, optional
        Name(s) of Orca tables containing data about alternatives. The first table is the
        primary one. Any additional tables need to have merge relationships ("broadcasts")
        specified so that they can be merged unambiguously onto the first table. The index
        of the primary table should be a unique ID. In this template, the 'choosers' and
        'alternatives' parameters replace the 'tables' parameter. Both are required for
        fitting a model, but do not have to be provided when the object is created.
        Reserved column names: 'chosen'.

    model_expression : str, optional
        Patsy-style right-hand-side model expression representing the utility of a
        single alternative. Passed to `choicemodels.MultinomialLogit()`. This parameter
        is required for fitting a model, but does not have to be provided when the object
        is created.

    choice_column : str, optional
        Name of the column indicating observed choices, for model estimation. The column
        should contain integers matching the id of the primary `alternatives` table. This
        parameter is required for fitting a model, but it does not have to be provided
        when the object is created. Not required for simulation.

    chooser_filters : str or list of str, optional
        Filters to apply to the chooser data before fitting the model. These are passed to
        `pd.DataFrame.query()`. Filters are applied after any additional tables are merged
        onto the primary one. Replaces the `fit_filters` argument in UrbanSim.

    chooser_sample_size : int, optional
        Number of choosers to sample, for faster model fitting. Sampling is random and may
        vary between model runs.

    alt_filters : str or list of str, optional
        Filters to apply to the alternatives data before fitting the model. These are
        passed to `pd.DataFrame.query()`. Filters are applied after any additional tables
        are merged onto the primary one. Replaces the `fit_filters` argument in UrbanSim.
        Choosers whose chosen alternative is removed by these filters will not be included
        in the model estimation.

    alt_sample_size : int, optional
        Numer of alternatives to sample for each choice scenario. For now, only random
        sampling is supported. If this parameter is not provided, we will use a sample
        size of one less than the total number of alternatives. (ChoiceModels codebase
        currently requires sampling.) The same sample size is used for estimation and
        prediction.

    out_choosers : str or list of str, optional
        Name(s) of Orca tables to draw choice scenario data from, for simulation. If not
        provided, the `choosers` parameter will be used. Same guidance applies. Reserved
        column names: 'chosen', 'join_index', 'observation_id'.

    out_alternatives : str or list of str, optional
        Name(s) of Orca tables containing data about alternatives, for simulation. If not
        provided, the `alternatives` parameter will be used. Same guidance applies.
        Reserved column names: 'chosen', 'join_index', 'observation_id'.

    out_column : str, optional
        Name of the column to write simulated choices to. If it does not already exist
        in the primary `out_choosers` table, it will be created. If not provided, the
        `choice_column` will be used. If the column already exists, choices will be cast
        to match its data type. If the column is generated on the fly, it will be given
        the same data type as the index of the alternatives table. Replaces the
        `out_fname` argument in UrbanSim.

    out_chooser_filters : str or list of str, optional
        Filters to apply to the chooser data before simulation. If not provided, no
        filters will be applied. Replaces the `predict_filters` argument in UrbanSim.

    out_alt_filters : str or list of str, optional
        Filters to apply to the alternatives data before simulation. If not provided, no
        filters will be applied. Replaces the `predict_filters` argument in UrbanSim.

    constrained_choices : bool, optional
        "True" means alternatives have limited capacity. "False" (default) means that
        alternatives can accommodate an unlimited number of choosers.

    alt_capacity : str, optional
        Name of a column in the out_alternatives table that expresses the capacity of
        alternatives. If not provided and constrained_choices is True, each alternative
        is interpreted as accommodating a single chooser.

    chooser_size : str, optional
        Name of a column in the out_choosers table that expresses the size of choosers.
        Choosers might have varying sizes if the alternative capacities are amounts
        rather than counts -- e.g. square footage. Chooser sizes must be in the same units
        as alternative capacities. If not provided and constrained_choices is True, each
        chooser has a size of 1.

    max_iter : int or None, optional
        Maximum number of choice simulation iterations. If None (default), the algorithm
        will iterate until all choosers are matched or no alternatives remain.

    name : str, optional
        Name of the model step, passed to ModelManager. If none is provided, a name is
        generated each time the `fit()` method runs.

    tags : list of str, optional
        Tags, passed to ModelManager.

    Attributes
    ----------
    All parameters can also be get and set as properties. The following attributes should
    be treated as read-only.

    choices : pd.Series
        Available after the model step is run. List of chosen alternative id's, indexed
        with the chooser id. Does not persist when the model step is reloaded from
        storage.

    mergedchoicetable : choicemodels.tools.MergedChoiceTable
        Table built for estimation or simulation. Does not persist when the model step is
        reloaded from storage. Not available if choices have capacity constraints,
        because multiple choice tables are generated iteratively.

    model : choicemodels.MultinomialLogitResults
        Available after a model has been fit. Persists when reloaded from storage.

    probabilities : pd.Series
        Available after the model step is run -- but not if choices have capacity
        constraints, which requires probabilities to be calculated multiple times.
        Provides list of probabilities corresponding to the sampled alternatives, indexed
        with the chooser and alternative id's. Does not persist when the model step is
        reloaded from storage.

    """

    def __init__(self, choosers=None, alternatives=None, model_expression=None,
                 choice_column=None, chooser_filters=None, chooser_sample_size=None,
                 alt_filters=None, alt_sample_size=None, out_choosers=None,
                 out_alternatives=None, out_column=None, out_chooser_filters=None,
                 out_alt_filters=None, constrained_choices=False, alt_capacity=None,
                 chooser_size=None, max_iter=None, mct_intx_ops=None, name=None, tags=[]):

        self._listeners = []

        # Parent class can initialize the standard parameters
        TemplateStep.__init__(self, tables=None, model_expression=model_expression,
                              filters=None, out_tables=None, out_column=out_column, out_transform=None,
                              out_filters=None, name=name, tags=tags)

        # Custom parameters not in parent class
        self.choosers = choosers
        self.alternatives = alternatives
        self.choice_column = choice_column
        self.chooser_filters = chooser_filters
        self.chooser_sample_size = chooser_sample_size
        self.alt_filters = alt_filters
        self.alt_sample_size = alt_sample_size
        self.out_choosers = out_choosers
        self.out_alternatives = out_alternatives
        self.out_chooser_filters = out_chooser_filters
        self.out_alt_filters = out_alt_filters
        self.constrained_choices = constrained_choices
        self.alt_capacity = alt_capacity
        self.chooser_size = chooser_size
        self.max_iter = max_iter
        self.mct_intx_ops = mct_intx_ops

        # Placeholders for model fit data, filled in by fit() or from_dict()
        self.summary_table = None
        self.fitted_parameters = None
        self.model = None

        # Placeholders for diagnostic data, filled in by fit() or run()
        self.mergedchoicetable = None
        self.probabilities = None
        self.choices = None



    def bind_to(self, callback):
        self._listeners.append(callback)

    def send_to_listeners(self, param, value):
        for callback in self._listeners:
            callback(param, value)

    @classmethod
    def from_dict(cls, d):
        """
        Create an object instance from a saved dictionary representation.

        Parameters
        ----------
        d : dict

        Returns
        -------
        LargeMultinomialLogitStep

        """
        check_choicemodels_version()
        from choicemodels import MultinomialLogitResults

        # Pass values from the dictionary to the __init__() method
        obj = cls(choosers=d['choosers'], alternatives=d['alternatives'],
                  model_expression=d['model_expression'], choice_column=d['choice_column'],
                  chooser_filters=d['chooser_filters'],
                  chooser_sample_size=d['chooser_sample_size'],
                  alt_filters=d['alt_filters'], alt_sample_size=d['alt_sample_size'],
                  out_choosers=d['out_choosers'], out_alternatives=d['out_alternatives'],
                  out_column=d['out_column'], out_chooser_filters=d['out_chooser_filters'],
                  out_alt_filters=d['out_alt_filters'],
                  constrained_choices=d['constrained_choices'], alt_capacity=d['alt_capacity'],
                  chooser_size=d['chooser_size'], max_iter=d['max_iter'],
                  mct_intx_ops=d.get('mct_intx_ops', None), name=d['name'],
                  tags=d['tags'])

        # Load model fit data
        obj.summary_table = d['summary_table']
        obj.fitted_parameters = d['fitted_parameters']

        if obj.fitted_parameters is not None:
            obj.model = MultinomialLogitResults(model_expression=obj.model_expression,
                                                fitted_parameters=obj.fitted_parameters)

        return obj

    def to_dict(self):
        """
        Create a dictionary representation of the object.

        Returns
        -------
        dict

        """
        d = {
            'template': self.template,
            'template_version': self.template_version,
            'name': self.name,
            'tags': self.tags,
            'choosers': self.choosers,
            'alternatives': self.alternatives,
            'model_expression': self.model_expression,
            'choice_column': self.choice_column,
            'chooser_filters': self.chooser_filters,
            'chooser_sample_size': self.chooser_sample_size,
            'alt_filters': self.alt_filters,
            'alt_sample_size': self.alt_sample_size,
            'out_choosers': self.out_choosers,
            'out_alternatives': self.out_alternatives,
            'out_column': self.out_column,
            'out_chooser_filters': self.out_chooser_filters,
            'out_alt_filters': self.out_alt_filters,
            'constrained_choices': self.constrained_choices,
            'alt_capacity': self.alt_capacity,
            'chooser_size': self.chooser_size,
            'max_iter': self.max_iter,
            'mct_intx_ops': self.mct_intx_ops,
            'summary_table': self.summary_table,
            'fitted_parameters': self.fitted_parameters,
        }
        return d

    # TO DO - there has got to be a less verbose way to handle getting and setting

    @property
    def choosers(self):
        return self.__choosers

    @choosers.setter
    def choosers(self, value):
        self.__choosers = self._normalize_table_param(value)
        self.send_to_listeners('choosers', value)

    @property
    def alternatives(self):
        return self.__alternatives

    @alternatives.setter
    def alternatives(self, value):
        self.__alternatives = self._normalize_table_param(value)
        self.send_to_listeners('alternatives', value)

    @property
    def model_expression(self):
        return self.__model_expression

    @model_expression.setter
    def model_expression(self, value):
        self.__model_expression = value
        self.send_to_listeners('model_expression', value)

    @property
    def choice_column(self):
        return self.__choice_column

    @choice_column.setter
    def choice_column(self, value):
        self.__choice_column = value
        self.send_to_listeners('choice_column', value)

    @property
    def chooser_filters(self):
        return self.__chooser_filters

    @chooser_filters.setter
    def chooser_filters(self, value):
        self.__chooser_filters = value
        self.send_to_listeners('chooser_filters', value)

    @property
    def chooser_sample_size(self):
        return self.__chooser_sample_size

    @chooser_sample_size.setter
    def chooser_sample_size(self, value):
        self.__chooser_sample_size = value
        self.send_to_listeners('chooser_sample_size', value)

    @property
    def alt_filters(self):
        return self.__alt_filters

    @alt_filters.setter
    def alt_filters(self, value):
        self.__alt_filters = value
        self.send_to_listeners('alt_filters', value)

    @property
    def alt_sample_size(self):
        return self.__alt_sample_size

    @alt_sample_size.setter
    def alt_sample_size(self, value):
        self.__alt_sample_size = value
        self.send_to_listeners('alt_sample_size', value)

    @property
    def out_choosers(self):
        return self.__out_choosers

    @out_choosers.setter
    def out_choosers(self, value):
        self.__out_choosers = self._normalize_table_param(value)
        self.send_to_listeners('out_choosers', value)

    @property
    def out_alternatives(self):
        return self.__out_alternatives

    @out_alternatives.setter
    def out_alternatives(self, value):
        self.__out_alternatives = self._normalize_table_param(value)
        self.send_to_listeners('out_alternatives', value)

    @property
    def out_column(self):
        return self.__out_column

    @out_column.setter
    def out_column(self, value):
        self.__out_column = value
        self.send_to_listeners('out_column', value)

    @property
    def out_chooser_filters(self):
        return self.__out_chooser_filters

    @out_chooser_filters.setter
    def out_chooser_filters(self, value):
        self.__out_chooser_filters = value
        self.send_to_listeners('out_chooser_filters', value)

    @property
    def out_alt_filters(self):
        return self.__out_alt_filters

    @out_alt_filters.setter
    def out_alt_filters(self, value):
        self.__out_alt_filters = value
        self.send_to_listeners('out_alt_filters', value)

    @property
    def constrained_choices(self):
        return self.__constrained_choices

    @constrained_choices.setter
    def constrained_choices(self, value):
        self.__constrained_choices = value
        self.send_to_listeners('constrained_choices', value)

    @property
    def alt_capacity(self):
        return self.__alt_capacity

    @alt_capacity.setter
    def alt_capacity(self, value):
        self.__alt_capacity = value
        self.send_to_listeners('alt_capacity', value)

    @property
    def chooser_size(self):
        return self.__chooser_size

    @chooser_size.setter
    def chooser_size(self, value):
        self.__chooser_size = value
        self.send_to_listeners('chooser_size', value)

    @property
    def max_iter(self):
        return self.__max_iter

    @max_iter.setter
    def max_iter(self, value):
        self.__max_iter = value
        self.send_to_listeners('max_iter', value)

    @property
    def mct_intx_ops(self):
        return self.__mct_intx_ops

    @mct_intx_ops.setter
    def mct_intx_ops(self, value):
        self.__mct_intx_ops = value
        self.send_to_listeners('mct_intx_ops', value)

    def run(self, chooser_batch_size=None, interaction_terms=None):
        """
        Run the model step: simulate choices and use them to update an Orca column.

        The simulated choices are saved to the class object for diagnostics. If choices
        are unconstrained, the choice table and the probabilities of sampled alternatives
        are saved as well.

        Parameters
        ----------
        chooser_batch_size : int
            This parameter gets passed to
            choicemodels.tools.simulation.iterative_lottery_choices and is a temporary
            workaround for dealing with memory issues that arise from generating massive
            merged choice tables for simulations that involve large numbers of choosers,
            large numbers of alternatives, and large numbers of predictors. It allows the
            user to specify a batch size for simulating choices one chunk at a time.

        interaction_terms : pandas.Series, pandas.DataFrame, or list of either, optional
            Additional column(s) of interaction terms whose values depend on the
            combination of observation and alternative, to be merged onto the final data
            table. If passed as a Series or DataFrame, it should include a two-level
            MultiIndex. One level's name and values should match an index or column from
            the observations table, and the other should match an index or column from the
            alternatives table.

        Returns
        -------
        None

        """

        # Clear simulation attributes from the class object
        self.mergedchoicetable = None
        self.probabilities = None
        self.choices = None

        if interaction_terms is not None:
            uniq_intx_idx_names = set([
                idx for intx in interaction_terms for idx in intx.index.names])
            obs_extra_cols = to_list(self.chooser_size) + \
                list(uniq_intx_idx_names)
            alts_extra_cols = to_list(
                self.alt_capacity) + list(uniq_intx_idx_names)

        else:
            obs_extra_cols = to_list(self.chooser_size)
            alts_extra_cols = to_list(self.alt_capacity)

        observations = orca.get_table(self.out_choosers).to_frame()
        agents = observations.values
        agent_idx = observations.index.values
        if len(observations) == 0:
            print("No valid choosers")
            return

        alternatives = orca.get_table(self.out_alternatives).to_frame()
        if len(alternatives) == 0:
            print("No valid alternatives")
            return
        alts_col = [col for col in alternatives.columns if col not in [self.alt_capacity]]
        alts = np.repeat(alternatives[alts_col].values, alternatives[self.alt_capacity].values, axis=0)
        alt_idx = np.repeat(alternatives.index.values, alternatives[self.alt_capacity].values, axis=0)

		# given observations, alternatives
		# produce choice
        # agents: agents     index: agent_idx
        # alternatives: alts index: alt_idx
        ##
        # max matrix size set to reduce the matrix size for faster speed
        max_matrix_size = 1000
        n = agents.shape[0]
        m = alts.shape[0]

        # initialized HU index for the geo
        hu_index = np.arange(m)
        out_index = geo_slice.index
        # initialize start index for HH
        start_ind = 0
        # debug
        # init_assigned_hh = (out.loc[geo_slice.index, 'matched_household_id'] != -1).sum()
        while hu_index.shape[0] > 0:
            # if #of hh exceed max size, create chunk with size min(n, max_matrix_size)
            num_hu = min(m, max_matrix_size)
            num_hh = min(n, max_matrix_size)
            # if index exceed range, use n
            end_ind = min(start_ind + num_hh, n)
            hh_sample_ind = np.arange(start_ind, end_ind, dtype=np.int64)
            # get the HH split by index
            hh_split_i = observations[start_ind:end_ind, :]
            # get group by bg
            hu_bg_unique, hu_bg_unique_count = np.unique(alternatives[hu_index][:, [1, 3, 4]], axis=0, return_counts=True)
            if len(hu_bg_unique_count) > 1:
                # if more than one bg in this geo
                # generate importance score based on syn HH summary table by BG
                bg = np.array([syn_bg_count[np.all(np.isin(syn_bg_count[:, :3], ar), axis=1)][0]
                            # if the bg is found in the synthesis count
                        if np.all(np.isin(syn_bg_count[:, :3], ar), axis=1).sum()!=0
                            # if not found, use count 1
                        else np.append(ar, 1)
                        for ar in hu_bg_unique])
                # bg = syn_bg_count[np.all(np.isin(syn_bg_count[:, :3], hu_bg_unique), axis=1)]
                bg_count = np.repeat(bg[:, -1], hu_bg_unique_count) if len(bg) > 0 else np.array([])
                p = np.divide(bg_count, np.sum(bg_count)) if len(bg_count) > 0 else None
            else:
                # if only one bg in this geo, skip this step
                p = None

            # get sampled HU split by random sampling
            hu_sample_ind = np.random.choice(m, num_hu, replace=False, p=p)

            # not including the last col(matched_hh_id)
            hu_split_i = alternatives[hu_index][hu_sample_ind, :-1]

            # run placement algo on the set of Building and HH
            # building_result list: ind=building_id value=hh_id

            match_result, match_building_ind = matching(hu_split_i, hh_split_i)
            # update buildings with match hh_id
            # TODO: updating matching result
            hu_ind_to_update = out_index[hu_index[match_building_ind]]
            # out.loc[hu_ind_to_update, 'matched_household_id'] = [
            #     observations[hh_sample_ind[hh_ind], 0] for hh_ind in match_result]

            ## important
            # remove assigned hu from the pool
            hu_index = np.delete(hu_index, match_building_ind)

            # current_geo_slice = alternatives[[hu_index], :]
            m = hu_index.shape[0]

            # update start_ind
            start_ind += num_hh
            # if next start ind > hh length, stop while loop
            if start_ind >= n:
                # debug
                break

        # Save choices to class object for diagnostics
        self.choices = choices

        # TODO: Update this
        # Update Orca
        # update_column(table=self.out_choosers,
        #               fallback_table=self.choosers,
        #               column=self.out_column,
        #               fallback_column=self.choice_column,
        #               data=choices)

#credits goes to Geekforgeek
@numba.jit(nopython=True)
def wPrefersM1OverM(prefer, w, m, m1, N):
    for i in range(N):
        if (prefer[w][i] == m1):
            return True
        if (prefer[w][i] == m):
            return False

@numba.jit(nopython=True)
def stable_matching(man_prefer, wmn_prefer):
    # return list(ind=wmn, value=man)
    N = len(man_prefer)
    wPartner = [-1 for i in range(N)]
    mFree = [False for i in range(N)]

    freeCount = N
    while (freeCount > 0):
        m = 0
        while (m < N):
            if (mFree[m] == False):
                break
            m += 1

        i = 0
        while i < N and mFree[m] == False:
            w = man_prefer[m][i]
            if wPartner[w] == -1:
                wPartner[w] = m
                mFree[m] = True
                freeCount -= 1
            else:
                m1 = wPartner[w]
                if (wPrefersM1OverM(wmn_prefer, w, m, m1, N) == False):
                    wPartner[w] = m
                    mFree[m] = True
                    mFree[m1] = False
            i += 1
    return wPartner

def row_rank( match_matrix, row):
    '''
    Row ranking

    args:
    match_matrix:   np.ndarray matrix to ranking against
    row:            np.array row

    return:
    prefer_ranking  np.array
    '''
    weight = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], dtype=np.int64)
    # bool match geo
    scores_mat = np.zeros((match_matrix.shape[0], 8), dtype=np.int64)
    scores_mat[:, 0]    = ((match_matrix[:, 1:5] != row[1:5])       * weight[0:4]).sum(axis=1)
    # bool tanure, building_type
    scores_mat[:, 1: 3] = (match_matrix[:, 5:7] != row[5:7])         * weight[4:6]
    # numeric num_of_unit, property_values/rent, year_built, and income/biv
    scores_mat[:, 3:8]  = np.abs(match_matrix[:, 7:12] - row[7:12]) * weight[6:11]
    return scores_mat.sum(axis=1).argsort()

def matching(alts, agents):
    '''
	Transformed from forecast_data_input/placement.py:357 placement
    solving the stable matching program given encoded buildings and hh
    Input:
        - Alts record matrix: np.ndarray
        - Agents record matrix: np.ndarray
    Return:
        - matching result
    '''
    # # of agents rows
    n = agents.shape[0]
    # # of alts rows
    m =  alts.shape[0]
    # calculate alts prefer matrix and agents prefer matrix
    agent_prefer_rank = np.zeros((n, m))
    i = 0
    for hh in agents:
        agent_prefer_rank[i, :] = row_rank( alts, hh)
        i += 1
    alt_prefer_rank = np.zeros((m, n))
    i = 0
    for b in  alts:
        alt_prefer_rank[i, :] = row_rank(agents, b)
        i += 1
    if n < m:
        # Adding alts dummies
        dummies_for_alts = np.repeat(np.arange(n, m), np.array([m])).reshape(
            (m-n, m)).transpose()
        # append dummies
        alt_prefer_rank = np.hstack((alt_prefer_rank, dummies_for_alts))
        dummies_for_agents = np.repeat(
            np.arange(m), np.array([m-n])).reshape((m, m-n)).transpose()
        # append dummies
        agent_prefer_rank = np.vstack((agent_prefer_rank, dummies_for_agents))
    elif n > m:
        # Adding agents dummies
        dummies_for_agents = np.repeat(np.arange(m, n), np.array([n])).reshape(
            (n-m, n)).transpose()
        # append dummies
        agent_prefer_rank = np.hstack((agent_prefer_rank, dummies_for_agents))
        dummies_for_alts = np.repeat(
            np.arange(n), np.array([n-m])).reshape((n, n-m)).transpose()
        # append dummies
        alt_prefer_rank = np.vstack((alt_prefer_rank, dummies_for_alts))
    # take min(n, m) count as final result
    result = stable_matching(agent_prefer_rank.astype(np.int64), alt_prefer_rank.astype(np.int64))
    # only keep l result excluding dummies
    l = min(n, m)
    if n > m:
        # if more agents presented, take the first m record, all alts have been assigned
        matched_alt_id = [i for i in range(m)]
        result_wo_dum = np.array(result[0: l], dtype=np.int64)
    else:
        # if more or equal alts, take the first n record, all agents have been assigned
        # get the alts which have valid agents assigned
        # init the list with dtype int
        matched_alt_id = [1]
        # remove 1 from list
        matched_alt_id.pop()
        for i, hh_ind in enumerate(result):
            if hh_ind in range(n):
                matched_alt_id.append(i)
        result_wo_dum = np.array([result[i]
                                  for i in matched_alt_id], dtype=np.int64)
    return (
        result_wo_dum,
        np.array(matched_alt_id, dtype=np.int64)
    )
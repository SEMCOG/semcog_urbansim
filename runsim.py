import numpy as np
import statsmodels.formula.api as smf
from patsy import dmatrices


def apply_filter_query(df, filters):
    query = ' and '.join(filters)
    return df.query(query)


def estimate(segment, estimate_filters, model_expression):
    segment = apply_filter_query(segment, estimate_filters)
    model = smf.ols(formula=model_expression, data=segment)
    return model.fit()


def simulate(segment, simulate_filters, model_expression, model_params,
             yexp=False):
    segment = apply_filter_query(segment, simulate_filters)
    _, est_data = dmatrices(model_expression, segment, return_type='dataframe')
    sim_data = est_data.dot(model_params)
    if yexp:
        sim_data = sim_data.apply(np.exp)
    return sim_data


def run_year(table, est_filters, sim_filters, model_exp, segmentation_col,
             update_col, year=None, yexp=False):
    segments = table.groupby(segmentation_col)

    for seg_name, seg in segments:
        model_fit = estimate(seg, est_filters, model_exp)
        update = simulate(
            seg, sim_filters, model_exp, model_fit.params, yexp)
        table[update_col][update.index] = update

    return table

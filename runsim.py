import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def apply_filter_query(df, filters):
    """
    Use the DataFrame.query method to filter a table down to the
    desired rows.

    Parameters
    ----------
    df : pandas.DataFrame
    filters : list of str
        List of filters to apply. Will be joined together with
        ' and ' and passed to DataFrame.query.

    Returns
    -------
    filtered_df : pandas.DataFrame

    """
    query = ' and '.join(filters)
    return df.query(query)


def fit_model(df, filters, model_expression):
    """
    Use statsmodels to construct a model relation.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to use for fit. Should contain all the columns
        referenced in the `model_expression`.
    filters : list of str
        Any filters to apply before doing the model fit.
    model_expression : str
        A patsy model expression that can be used with statsmodels.
        Should contain both the left- and right-hand sides.

    Returns
    -------
    fit : statsmodels.regression.linear_model.OLSResults

    """
    df = apply_filter_query(df, filters)
    model = smf.ols(formula=model_expression, data=df)
    return model.fit()


def predict(df, filters, model_fit, ytransform=None):
    """
    Apply model to new data to predict new dependent values.

    Parameters
    ----------
    df : pandas.DataFrame
    filters : list of str
        Any filters to apply before doing prediction.
    model_fit : statsmodels.regression.linear_model.OLSResults
        Result of model estimation.
    ytransform : callable, optional
        A function to call on the array of predicted output.
        For example, if the model relation is predicting the log
        of price, you might pass ``ytransform=np.exp`` so that
        the results reflect actual price.

        By default no transformation is applied.

    Returns
    -------
    result : pandas.Series
        Predicted values as a pandas Series. Will have the index of `df`
        after applying filters.

    """
    df = apply_filter_query(df, filters)
    sim_data = model_fit.predict(df)
    if ytransform:
        sim_data = ytransform(sim_data)
    return pd.Series(sim_data, index=df.index)


def try_run(table, est_filters, sim_filters, model_exp, segmentation_col,
            update_col, year=None, ytransform=None):
    segments = table.groupby(segmentation_col)

    for seg_name, seg in segments:
        model_fit = fit_model(seg, est_filters, model_exp)
        update = predict(seg, sim_filters, model_fit, ytransform)
        table[update_col][update.index] = update

    return table

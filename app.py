# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, dash_table, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

import numpy as np
import os
import copy
import sys
import numbers
from flask_caching import Cache
import time


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
			query = ' and '.join(filters)
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
	if name.endswith('_min'):
		name = name[:-4]
		comp = '>='
	elif name.endswith('_max'):
		name = name[:-4]
		comp = '<'
	else:
		comp = '=='

	result = '{} {} {!r}'.format(name, comp, value)
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

	filters = [_filterize(name, val)
				for name, val in filter_series.iteritems()
				if not (name in ignore or
						(isinstance(val, numbers.Number) and
						np.isnan(val)))]

	return apply_filter_query(table, filters)

run_folder = "/home/da/semcog_urbansim/runs"
run_num = "run336"

hdf = pd.HDFStore(os.path.join(run_folder, '%s.h5'%run_num), 'r')

p_base = hdf["/base/parcels"]
mcd_total = pd.read_csv('/home/da/share/urbansim/RDF2050/model_inputs/base_tables/mcd_totals_2020_2050_nov15.csv', index_col='mcd')
# semmcds = hdf['/base/semmcds']
# semmcds = semmcds.reset_index()[['semmcd_id', 'large_area_id']].set_index('semmcd_id')
semmcds = p_base[['city_id', 'large_area_id']].groupby(['city_id', 'large_area_id']).count().reset_index().set_index('city_id')

usecache = True
if usecache == False:
	print("running hh_by_mcd_year")
	hh_by_mcd_year = pd.DataFrame(index=p_base.city_id.unique()).sort_index()
	hu_by_mcd_year = pd.DataFrame(index=p_base.city_id.unique()).sort_index()
	for year in range(2020, 2051):
		if "/%s/parcels" % year not in hdf.keys():
			break
		p = hdf["/%s/parcels" % year]
		b = hdf["/%s/buildings" % year]
		hh = hdf["/%s/households" % year]
		b = b.join(p.city_id, on='parcel_id')
		hh = hh.join(b.city_id, on='building_id')
		hh_vcount = hh.city_id.fillna(-1).astype(int).value_counts()
		hu_vcount = b[['city_id', 'residential_units']].groupby('city_id').sum()['residential_units']
		hh_by_mcd_year.loc[:, str(year)] = hh_vcount
		hu_by_mcd_year.loc[:, str(year)] = hu_vcount
	hh_by_mcd_year.to_csv('~/semcog_urbansim/data/cache/hh_by_cityid_year_%s.csv' % run_num)
	hu_by_mcd_year.to_csv('~/semcog_urbansim/data/cache/hu_by_cityid_year_%s.csv' % run_num)
else:
    hh_by_mcd_year = pd.read_csv('~/semcog_urbansim/data/cache/hh_by_cityid_year_%s.csv' % run_num, index_col=0)
    hu_by_mcd_year = pd.read_csv('~/semcog_urbansim/data/cache/hu_by_cityid_year_%s.csv' % run_num, index_col=0)

# hh_by_mcd_year['large_area_id'] = semmcds


# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Households by mcd, '+run_num),
	html.Div([
		'Select large_area_id',
		dcc.Dropdown( semmcds.large_area_id.unique(), '-', id='dropdown'),
	], style={"width": "400px"}),
	html.Div(
		id="charts",
		# children=[dcc.Graph(id=str(mcd), figure=px.line(pd.DataFrame({"mcd_total": mcd_total.loc[mcd], "simulated":hh_by_mcd_year.loc[mcd]}), title=mcd)) for mcd, row in mcd_total.iloc[:].iterrows()]
		children=[]
	)
])

CACHE_CONFIG = {
    'CACHE_TYPE': 'FileSystemCache',
	'CACHE_DIR' : 'data/cache',
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)


# perform expensive computations in this "global store"
# these computations are cached in a globally available
# redis memory store which is available across processes
# and for all time.
@cache.memoize()
def global_store(value):
    # simulate expensive query
    print(f'Computing value with {value}')
    time.sleep(3)
    return hh_by_mcd_year[hh_by_mcd_year.index.isin(semmcds[semmcds.large_area_id==value].index)], hu_by_mcd_year[hu_by_mcd_year.index.isin(semmcds[semmcds.large_area_id==value].index)]

@app.callback(
    Output(component_id='charts', component_property='children'),
    Input('dropdown', 'value')
)
def get_charts(la_id):
	print('run get_charts')
	simulated_hh, simulated_hu = global_store(la_id)
	mt = mcd_total.loc[mcd_total.index.isin(simulated_hh.index)]
	# print(simulated_hh)
	return [dcc.Graph(id=str(mcd), figure=px.line(
    pd.DataFrame({
        "mcd_total": mt.loc[mcd] if mcd in mt.index else pd.Series([],dtype=float),
        "simulated_hh": simulated_hh.loc[mcd],
        "simulated_hu": simulated_hu.loc[mcd],
    }), title=mcd, width=1100, height=450)) for mcd, _ in semmcds.loc[semmcds.large_area_id==la_id].iterrows()]

if __name__ == '__main__':
    app.run(debug=True, host= '192.168.185.65', port = 8080)

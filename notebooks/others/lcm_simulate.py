#!/usr/bin/env python
# coding: utf-8

# ### Imports and registering variables with orca 

# In[ ]:


import os
import numpy as np
import pandas as pd

import orca
from urbansim.utils import misc

import dataset
import variables
import models
import utils

orca.run(["build_networks"])
orca.run(["neighborhood_vars"])


# ## Simulation Example

# In[ ]:


# Simple transition model step
@orca.step('simple_jobs_transition')
def simple_transition_model(jobs):
    return utils.simple_transition(jobs, .005, 'building_id')

# Create the choice model object for sector_id 1 and register model as inectable
elcm1_model = utils.SimulationChoiceModel.from_yaml(str_or_buffer=misc.config('elcm/categories/elcm1.yaml'))
elcm1_model.set_simulation_params('elcm1', 'job_spaces','vacant_job_spaces', 'jobs', 'buildings')
orca.add_injectable('elcm1_model', elcm1_model)

# ELCM model step for sector 1
@orca.step('elcm1')
def choice_model_simulate(elcm1_model, jobs):
    choices = elcm1_model.simulate(choice_function=utils.unit_choices)

    print('There are %s unplaced agents.' % choices.isnull().sum())

    jobs.update_col_from_series(elcm1_model.choice_column, choices, cast=True)

    
# Simulate
orca.run(['simple_jobs_transition', 'elcm1'])


# ## Scoring Example

# In[ ]:


lids = orca.get_table('buildings').large_area_id.fillna(0).astype('int')

model = utils.SimulationChoiceModel.from_yaml(str_or_buffer=misc.config('hlcm/hlcm1.yaml'))
model.set_simulation_params('hlcm1', 'residential_units',
                            'vacant_residential_units', 'households', 'buildings', lids)

print('** Score is %s' % model.score())


# ## Plotting probabilities example

# In[ ]:


from bokeh.plotting import figure, output_notebook
output_notebook()

import datashader as ds
from datashader import transfer_functions as tf
from datashader.bokeh_ext import InteractiveImage
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select

def plot_by_high_low_proba(df):
    df['ones'] = 1
    region = x_range, y_range = ((df.x.min(),df.x.max()), (df.y.min(),df.y.max()))

    plot_width  = int(750)
    plot_height = int(plot_width//1.2)

    def base_plot(tools='pan,wheel_zoom,reset',plot_width=plot_width, plot_height=plot_height, **plot_args):
        p = figure(tools=tools, plot_width=plot_width, plot_height=plot_height,
            x_range=x_range, y_range=y_range, outline_line_color=None,
            min_border=0, min_border_left=0, min_border_right=0,
            min_border_top=0, min_border_bottom=0, **plot_args)

        p.axis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        return p

    options = dict(line_color=None, fill_color='blue', size=5)

    background = "black"
    export = partial(export_image, export_path="export", background=background)
    cm = partial(colormap_select, reverse=(background=="black"))

    def create_image(x_range, y_range, w=plot_width, h=plot_height):
        cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        color_key = color_key = {'low':'aqua', 'high':'red'}

        agg = cvs.points(df, 'x', 'y',  ds.count_cat('proba_class'))

        img = tf.shade(agg, color_key=color_key, how='eq_hist')
        return tf.dynspread(img, threshold=0.5, max_px=4)

    p = base_plot(background_fill_color=background)

    export(create_image(*region), 'probas')
    return InteractiveImage(p, create_image)


def elcm_probabilities(sector_id):
    model_name = 'elcm%s' % sector_id
    config_filename = 'elcm/categories/elcm%s.yaml' % sector_id

    model = utils.SimulationChoiceModel.from_yaml(str_or_buffer=misc.config(config_filename))

    model.set_simulation_params(model_name,
                                'job_spaces',
                                'vacant_job_spaces',
                                'jobs',
                                'buildings')

    choosers, alternatives = model.calculate_model_variables()

    return model.calculate_probabilities(choosers, alternatives)


def hlcm_probabilities(segment_id):
    model_name = 'hlcm%s' % segment_id
    config_filename = 'hlcm/hlcm%s.yaml' % segment_id

    model = utils.SimulationChoiceModel.from_yaml(str_or_buffer=misc.config(config_filename))

    model.set_simulation_params(model_name,
                                'residential_units',
                                'vacant_residential_units',
                                'households',
                                'buildings')

    choosers, alternatives = model.calculate_model_variables()

    return model.calculate_probabilities(choosers, alternatives)


def plot_building_level_probas(probas):
    buildings = orca.get_table('buildings')
    b_proba = pd.DataFrame({'proba':probas, 'x':buildings.parcels_centroid_x, 'y':buildings.parcels_centroid_y})

    b_proba = b_proba[(b_proba.x > 0) & (b_proba.y > 0) & (b_proba.x < 13640484) & (b_proba.y <  600000)]
    b_proba.proba = b_proba.proba*10000

    mean_proba = b_proba.proba.median()
    b_proba['proba_class'] = ''
    b_proba.proba_class[b_proba.proba > mean_proba] = 'high'
    b_proba.proba_class[b_proba.proba <= mean_proba] = 'low'
    b_proba.proba_class = b_proba.proba_class.astype('category')

    return plot_by_high_low_proba(b_proba)


def plot_elcm_probabilities(sector_id):
    probabilities = elcm_probabilities(sector_id)
    return plot_building_level_probas(probabilities)

def plot_hlcm_probabilities(segment_id):
    probabilities = hlcm_probabilities(segment_id)
    return plot_building_level_probas(probabilities)


# In[ ]:


plot_hlcm_probabilities(1)


# In[ ]:


plot_hlcm_probabilities(2)


# In[ ]:


plot_hlcm_probabilities(3)


# In[ ]:


plot_hlcm_probabilities(4)


# In[ ]:


plot_elcm_probabilities(1)


# In[ ]:


plot_elcm_probabilities(2)


# In[ ]:


plot_elcm_probabilities(3)


# In[ ]:


plot_elcm_probabilities(4)


# In[ ]:


plot_elcm_probabilities(5)


# In[ ]:


plot_elcm_probabilities(6)


# In[ ]:


plot_elcm_probabilities(7)


# In[ ]:


plot_elcm_probabilities(8)


# In[ ]:


plot_elcm_probabilities(9)


# In[ ]:


plot_elcm_probabilities(10)


# In[ ]:


plot_elcm_probabilities(11)


# In[ ]:


plot_elcm_probabilities(12)


# In[ ]:


plot_elcm_probabilities(13)


# In[ ]:


plot_elcm_probabilities(14)


# In[ ]:


plot_elcm_probabilities(15)


# In[ ]:


plot_elcm_probabilities(16)


# In[ ]:


plot_elcm_probabilities(17)


# In[ ]:


plot_elcm_probabilities(18)


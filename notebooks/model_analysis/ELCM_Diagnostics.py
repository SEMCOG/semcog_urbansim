#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize as opt
from urbansim.models import MNLDiscreteChoiceModel

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# ### Load model objects and saved explanatory variables

# In[2]:


buildings = pd.HDFStore('expvars.h5').expvars
buildings.sqft_price_nonres = buildings.sqft_price_nonres.fillna(0)

model_coeffs = {}
for model_config in os.listdir('configs/large_area_sector'):
    submodel = MNLDiscreteChoiceModel.from_yaml(
                    str_or_buffer='configs/large_area_sector/' + model_config)
    submodel_coeffs = submodel.fit_parameters.Coefficient
    large_area = int(submodel.alts_predict_filters.split(' == ')[-1])
    sector_id = (int(model_config.split('.')[0].split('elcm')[1]) - large_area) / 100000
    model_coeffs[(large_area, sector_id)] = submodel_coeffs


# ### Prep data

# In[3]:


buildings = buildings[buildings.non_residential_sqft > 0]
buildings['is_medical'] = buildings.building_type_id.isin([51, 52, 53]).astype('int')
buildings['is_office'] = buildings.building_type_id.isin([23, 24]).astype('int')
buildings['is_tcu'] = buildings.building_type_id.isin([33, 41, 42, 43]).astype('int')

building_cols = buildings.columns.values
sectoral_change = pd.read_csv('sectoral_change.csv').set_index('sector_id').sectoral_change

submodel_ids = [9, 4, 5, 6, 2, 
                16, 10, 17, 11, 14]

large_area_ids = [3, 5, 93, 99, 115, 125, 147, 161]

sectoral_changes = []
for submodel_id in submodel_ids:
    sectoral_changes.append(sectoral_change.loc[submodel_id])
    
x = np.transpose(buildings.as_matrix())

sectoral_changes = np.array(sectoral_changes).reshape((len(sectoral_changes), 1))
vacant_job_spaces = buildings.vacant_job_spaces.values
idx_medical = (buildings.is_medical == 1).values
idx_tcu = (buildings.is_tcu == 1).values

def get_submodel_coeffs(large_area_id):
    w = []

    for submodel_id in submodel_ids:
        coeffs = model_coeffs[(large_area_id, submodel_id)]
        if submodel_id == 14:
            coeffs = coeffs.append(pd.Series({'is_medical':0.0}))

        if submodel_id == 6:
            coeffs = coeffs.append(pd.Series({'is_tcu':0.0}))

        coeff_vals = pd.Series(0.0, index=building_cols)
        coeff_vals.loc[coeffs.index] = coeffs.values
        w.append(coeff_vals.values)

    return np.array(w)


# ## Summed probabilities by building type

# In[4]:


def calculate_proba(w, x):
    utilities = np.dot(w, x)
    exp_utility = np.exp(utilities)

    sum_expu_across_submodels = exp_utility.sum(axis=1, keepdims=True)
    proba = exp_utility / sum_expu_across_submodels

    return proba


# In[6]:


# Get ELCM coefficients for large area 3
large_area_id = 3
w = get_submodel_coeffs(large_area_id)

# Calculate probabilities
proba = np.transpose(calculate_proba(w, x))
proba_df = pd.DataFrame(proba, columns = submodel_ids)

# Sum probabilities by building type
proba_df['building_type_id'] = buildings.building_type_id.values
summed_proba_btype = proba_df.groupby('building_type_id').sum()


# In[7]:


summed_proba_btype


# ### Summed probabilities by building type for sector_id 6

# In[8]:


summed_proba_btype[6].plot.bar()


# ### Summed probabilities by building type for sector_id 14

# In[26]:


summed_proba_btype[14].plot.bar()


# ### Model coefficients for sector 6, in large area 3

# In[9]:


model_coeffs[(3, 6)]


# ### Model coefficients for sector 14, in large area 3

# In[8]:


model_coeffs[(3, 14)]


# ### Run summed probability analysis for all large areas

# In[10]:


for large_area_id in large_area_ids:
    print(large_area_id)
    w = get_submodel_coeffs(large_area_id)
    proba = np.transpose(calculate_proba(w, x))

    proba_df = pd.DataFrame(proba, columns = submodel_ids)
    proba_df['building_type_id'] = buildings.building_type_id.values

    summed_proba_btype = proba_df.groupby('building_type_id').sum()
    print(summed_proba_btype)


# ## Summed capacity-weighted probabilities by building type

# In[12]:


def capacity_weighted_proba(w, x, alt_capacity):
    proba = calculate_proba(w, x)
    capacity_weighted_proba = proba * alt_capacity
    sum_capac_wproba_across_submodels = capacity_weighted_proba.sum(axis=1, keepdims=True)
    capacity_weighted_proba_normalized = capacity_weighted_proba / sum_capac_wproba_across_submodels
    
    return capacity_weighted_proba_normalized


# In[27]:


large_area_id = 3
w = get_submodel_coeffs(large_area_id)
capac_weighted_proba = np.transpose(capacity_weighted_proba(w, x, vacant_job_spaces))

proba_df = pd.DataFrame(capac_weighted_proba, columns=submodel_ids)
proba_df['building_type_id'] = buildings.building_type_id.values

summed_capac_weighted_proba_btype = proba_df.groupby('building_type_id').sum()


# In[28]:


summed_capac_weighted_proba_btype


# ### Summed capacity-weighted probabilities by building type for sector_id 6

# In[29]:


summed_capac_weighted_proba_btype[6].plot.bar()


# ### Summed capacity-weighted probabilities by building type for sector_id 14

# In[30]:


summed_capac_weighted_proba_btype[14].plot.bar()


# ### Run summed capacity-weighted probability analysis for all large areas

# In[13]:


for large_area_id in large_area_ids:
    print(large_area_id)
    w = get_submodel_coeffs(large_area_id)
    capac_weighted_proba = np.transpose(capacity_weighted_proba(w, x, vacant_job_spaces))

    proba_df = pd.DataFrame(capac_weighted_proba, columns=submodel_ids)
    proba_df['building_type_id'] = buildings.building_type_id.values

    summed_capac_weighted_proba_btype = proba_df.groupby('building_type_id').sum()
    print(summed_capac_weighted_proba_btype)
    print('')


# ## What building type dummy coefficient value may result in closer fit between simulated / observed sector shares by building type

# In[15]:


# Objective function to minimize

def growth_share_deviation(coeff, w, x, obs_growth, idx_alts, growth_share_target, submodel_idx, coeff_idx, alt_capacity):
    w[submodel_idx, coeff_idx] = coeff
    capacity_weighted_proba_normalized = capacity_weighted_proba(w, x, alt_capacity)
    expected_growth = np.transpose(capacity_weighted_proba_normalized * obs_growth)
    
    alts_of_interest = expected_growth[idx_alts]
    growth_shares_by_submodel = alts_of_interest.sum(axis=0) / alts_of_interest.sum()
    abs_deviation = abs(growth_shares_by_submodel[submodel_idx] - growth_share_target)
    
    return abs_deviation


# In[16]:


w = get_submodel_coeffs(99)

coeffs = np.linspace(0, 20, 50)
errors = [growth_share_deviation(coeff, w, x, sectoral_changes, idx_medical, .44, -1, -3, vacant_job_spaces) for coeff in coeffs]
plt.plot(coeffs, errors)

coeffs = np.linspace(0, 20, 50)
errors = [growth_share_deviation(coeff, w, x, sectoral_changes, idx_tcu, .331, 3, -1, vacant_job_spaces) for coeff in coeffs]
plt.plot(coeffs, errors)


# In[18]:


w = get_submodel_coeffs(3)

coeffs = np.linspace(0, 50, 50)
errors = [growth_share_deviation(coeff, w, x, sectoral_changes, idx_medical, .44, -1, -3, vacant_job_spaces) for coeff in coeffs]
plt.plot(coeffs, errors)

coeffs = np.linspace(0, 50, 50)
errors = [growth_share_deviation(coeff, w, x, sectoral_changes, idx_tcu, .331, 3, -1, vacant_job_spaces) for coeff in coeffs]
plt.plot(coeffs, errors)


# In[19]:


for large_area_id in large_area_ids:
    print(large_area_id)
    w = get_submodel_coeffs(large_area_id)

    result = opt.minimize_scalar(growth_share_deviation, method='Brent', args=(w, x, sectoral_changes, idx_medical, .44, -1, -3, vacant_job_spaces))
    print(result.x)

    result = opt.minimize_scalar(growth_share_deviation, method='Brent', args=(w, x, sectoral_changes, idx_tcu, .331, 3, -1, vacant_job_spaces))
    print(result.x)

    print('')


# ## Suggested next steps

# X Turning off relocation
# 
# - * Sample small number of alts in ELCM/HLCM, instead of no-sampling
# 
# - * Region-wide estimation for ELCM/HLCM.  Still simulate by large area, but use the regionally-estimated coefficients.   Regionally by sector.   Regionally by income quartile.
# 
# 
# ###################################################################
# - Re-specify models with a greater focus on building type specifically (cross-sectional)
# 
# - Utility / probability analysis for submodels where summed probabilities overwhelmingly belong to one building type
# 
# - Try for better coefficient values by using different settings in estimation.  Try the non-convergence PR.  With and without regularization
# 
# - Paul's diffusion research:  sample fewer alternatives, try alternative samplers.   
# 
# - Calibrated building type dummies to base-year btype shares (longitudinal data) 

# In[ ]:





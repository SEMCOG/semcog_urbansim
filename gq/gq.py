import pandas as pd, numpy as np

# Function used to allocate aggregate GQ from large area controls to TAZ (to form the row marginal)
def random_choice(chooser_ids, alternative_ids, probabilities):
    choices = pd.Series([np.nan] * len(chooser_ids), index=chooser_ids)
    chosen = np.random.choice(
        alternative_ids, size=len(chooser_ids), replace=True, p=probabilities)
    choices[chooser_ids] = chosen
    return choices

tazcounts = pd.read_csv('tazcounts.csv').set_index('tazce10')
gqcontrols = pd.read_csv('largearea_gq_controls.csv')

# Configuration
max_iterations = 500
convergence_criteria = .000001
first_year_to_run = 2015

# Determine years to run from the control totals
years = []
for col in gqcontrols.columns:
    try: 
        if int(col) >= first_year_to_run:
            years.append(col)
    except:
        pass
    
for year in years:
    print year
    for lid in np.unique(gqcontrols.largearea_id):
        print 'Large area id %s' % lid
        gq_lid = gqcontrols[['age_grp',year]][gqcontrols.largearea_id==lid]
        gq_lid = gq_lid.set_index('age_grp')
        tazcounts_lid = tazcounts[['gq04','gq517','gq1834','gq3564','gq65plus']][tazcounts.largearea_id==lid]
        taz_sum = tazcounts_lid.gq04 + tazcounts_lid.gq517 + tazcounts_lid.gq1834 + tazcounts_lid.gq3564 + tazcounts_lid.gq65plus
        diff = gq_lid.sum().values[0] - taz_sum.sum()
        print 'GQ change is %s' % diff
        if diff != 0:
            ##Allocation of GQ total to TAZ to prepare the row marginal
            if diff > 0:
                taz_probs = taz_sum/taz_sum.sum()
                chosen_taz = random_choice(np.arange(diff), taz_probs.index.values, taz_probs.values)
                for taz in chosen_taz:
                    taz_sum[taz_sum.index.values==int(taz)]+=1
            if diff < 0:
                taz_probs = taz_sum/taz_sum.sum()
                chosen_taz = random_choice(np.arange(abs(diff)), taz_probs.index.values, taz_probs.values)
                for taz in chosen_taz:
                    taz_sum[taz_sum.index.values==int(taz)]-=1
            ##IPF
            marginal1 = taz_sum[taz_sum>0]
            marginal2 = gq_lid[year]
            tazes_to_fit = marginal1.index.values
            seed = tazcounts_lid[np.in1d(tazcounts_lid.index.values,tazes_to_fit)]
            i = 0
            while 1:
                i+=1
                axis1_adj = marginal1/seed.sum(axis=1)
                axis1_adj_mult = np.reshape(np.repeat(axis1_adj.values,seed.shape[1]),seed.shape)
                seed = seed*axis1_adj_mult
                axis2_adj = marginal2/seed.sum()
                axis2_adj_mult = np.tile(axis2_adj.values,len(seed)).reshape(seed.shape)
                seed = seed*axis2_adj_mult
                if ((np.abs(axis1_adj - 1).max() < convergence_criteria) and (np.abs(axis2_adj - 1).max() < convergence_criteria)) or (i >= max_iterations):
                    if i >= max_iterations:
                        print 'Terminated without convergence'
                    else:
                        print 'IPF converged! %s iterations' % i
                    rounded = np.round(seed)
                    rounding_error = marginal2.sum() - rounded.sum().sum()
                    ##TODO:  deal with rounding error in an intelligent way
                    print 'Total rounding error is %s' % rounding_error
                    for col in rounded.columns:
                        tazcounts.loc[rounded.index, col] = rounded[col].values
                    break
    tazcounts.to_csv('tazcounts%s.csv'%year)
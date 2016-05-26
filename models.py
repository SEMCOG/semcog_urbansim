import pandas as pd
import numpy as np
import time
import os
import random
from urbansim.models import transition, relocation
from urbansim.developer import sqftproforma, developer
from urbansim.utils import misc, networks
#import dataset, variables, utils, transcad
import dataset, variables, utils
import pandana as pdna

import orca


@orca.step()
def diagnostic(parcels,buildings,jobs,households,nodes,iter_var):
    parcels = parcels.to_frame()
    buildings = buildings.to_frame()
    jobs = jobs.to_frame()
    households = households.to_frame()
    nodes = nodes.to_frame()
    import pdb; pdb.set_trace()
    
@orca.step()
def rsh_estimate(buildings, nodes):
    return utils.hedonic_estimate("rsh.yaml", buildings, nodes)
    
@orca.step()
def rsh_simulate(buildings, nodes):
    return utils.hedonic_simulate("rsh.yaml", buildings, nodes,
                                  "sqft_price_res")
                                  
@orca.step()
def nrh_estimate(buildings, nodes):
    return utils.hedonic_estimate("nrh.yaml", buildings, nodes)

@orca.step()
def nrh_simulate(buildings, nodes):
    return utils.hedonic_simulate("nrh.yaml", buildings, nodes,
                                  "sqft_price_nonres")

@orca.step()
def hlcm_estimate(households, buildings, nodes):
    return utils.lcm_estimate("hlcm.yaml", households, "building_id",
                              buildings, nodes)

@orca.step()
def hlcm_simulate(households, buildings, nodes):
    return utils.lcm_simulate("hlcm.yaml", households, buildings, nodes,
                              "building_id", "residential_units",
                              "vacant_residential_units")
                            
@orca.step()
def elcm_estimate(jobs, buildings, nodes):
    return utils.lcm_estimate("elcm3.yaml", jobs, "building_id",
                              buildings, nodes)
    return utils.lcm_estimate("elcm5.yaml", jobs, "building_id",
                              buildings, nodes)
    return utils.lcm_estimate("elcm93.yaml", jobs, "building_id",
                              buildings, nodes)
    return utils.lcm_estimate("elcm99.yaml", jobs, "building_id",
                              buildings, nodes)
    return utils.lcm_estimate("elcm115.yaml", jobs, "building_id",
                              buildings, nodes)
    return utils.lcm_estimate("elcm125.yaml", jobs, "building_id",
                              buildings, nodes)
    return utils.lcm_estimate("elcm147.yaml", jobs, "building_id",
                              buildings, nodes)
    return utils.lcm_estimate("elcm161.yaml", jobs, "building_id",
                              buildings, nodes)
                            
@orca.step()
def elcm_simulate(jobs, buildings, nodes):
    jobs_df = jobs.to_frame()
    jobs3 = jobs_df[jobs_df.lid==3]
    jobs5 = jobs_df[jobs_df.lid==5]
    jobs93 = jobs_df[jobs_df.lid==93]
    jobs99 = jobs_df[jobs_df.lid==99]
    jobs115 = jobs_df[jobs_df.lid==115]
    jobs125 = jobs_df[jobs_df.lid==125]
    jobs147 = jobs_df[jobs_df.lid==147]
    jobs161 = jobs_df[jobs_df.lid==161]

    buildings = buildings.to_frame()
    buildings3 = buildings[buildings.large_area_id==3]
    buildings5 = buildings[buildings.large_area_id==5]
    buildings93 = buildings[buildings.large_area_id==93]
    buildings99 = buildings[buildings.large_area_id==99]
    buildings115 = buildings[buildings.large_area_id==115]
    buildings125 = buildings[buildings.large_area_id==125]
    buildings147 = buildings[buildings.large_area_id==147]
    buildings161 = buildings[buildings.large_area_id==161]
    
    def register_broadcast_simulate_segment(jobs_df_name, jobs_df, buildings_df_name, buildings_df, yaml_name):
        orca.add_table(jobs_df_name,jobs_df)
        orca.add_table(buildings_df_name,buildings_df)
        orca.broadcast('nodes', buildings_df_name, cast_index=True, onto_on='_node_id')
        orca.broadcast('parcels', buildings_df_name, cast_index=True, onto_on='parcel_id')
        orca.broadcast(buildings_df_name, jobs_df_name, cast_index=True, onto_on='building_id')
        jobs_df = orca.get_table(jobs_df_name)
        buildings_df = orca.get_table(buildings_df_name)
        utils.lcm_simulate(yaml_name, jobs_df, buildings_df, nodes,
                                  "building_id", "job_spaces",
                                  "vacant_job_spaces")
        jobs.update_col_from_series('building_id',
                                    pd.Series(jobs_df.building_id.values, index=jobs_df.index.values))
    control_segments = (['jobs3',jobs3,'buildings3',buildings3,'elcm3.yaml'],
                        ['jobs5',jobs5,'buildings5',buildings5,'elcm5.yaml'],
                        ['jobs93',jobs93,'buildings93',buildings93,'elcm93.yaml'],
                        ['jobs99',jobs99,'buildings99',buildings99,'elcm99.yaml'],
                        ['jobs115',jobs115,'buildings115',buildings115,'elcm115.yaml'],
                        ['jobs125',jobs125,'buildings125',buildings125,'elcm125.yaml'],
                        ['jobs147',jobs147,'buildings147',buildings147,'elcm147.yaml'],
                        ['jobs161',jobs161,'buildings161',buildings161,'elcm161.yaml'],)
    for csegment in control_segments:
        register_broadcast_simulate_segment(csegment[0],csegment[1],csegment[2],csegment[3],csegment[4])

@orca.step()
def households_relocation(households, annual_relocation_rates_for_households):
    relocation_rates = annual_relocation_rates_for_households.to_frame()
    relocation_rates = relocation_rates.rename(columns={'age_max': 'age_of_head_max', 'age_min': 'age_of_head_min'})
    relocation_rates.probability_of_relocating = relocation_rates.probability_of_relocating*.05
    reloc = relocation.RelocationModel(relocation_rates, 'probability_of_relocating')
    hh = households.to_frame(households.local_columns)
    idx_reloc = reloc.find_movers(hh)
    households.update_col_from_series('building_id',
                                    pd.Series(-1, index=idx_reloc))
    _print_number_unplaced(households, 'building_id')

@orca.step()
def jobs_relocation(jobs, annual_relocation_rates_for_jobs):
    relocation_rates = annual_relocation_rates_for_jobs.to_frame()
    relocation_rates.job_relocation_probability = relocation_rates.job_relocation_probability*.05
    reloc = relocation.RelocationModel(relocation_rates, 'job_relocation_probability')
    j = jobs.to_frame(jobs.local_columns)
    idx_reloc = reloc.find_movers(j)
    j.loc[idx_reloc, "building_id"] = -1
    orca.add_table("jobs", j)
    _print_number_unplaced(jobs, 'building_id')

@orca.step()
def households_transition(households, persons, annual_household_control_totals, iter_var):
    ct = annual_household_control_totals.to_frame()
    for col in ct.columns:
        i = 0
        if col.endswith('_max'):
            if len(ct[col][ct[col]==-1]) > 0:
                ct[col][ct[col]==-1] = np.inf
                i+=1
            if i > 0:
                ct[col] = ct[col] + 1
    tran = transition.TabularTotalsTransition(ct, 'total_number_of_households')
    model = transition.TransitionModel(tran)
    hh = households.to_frame(households.local_columns)
    p = persons.to_frame(persons.local_columns)
    new, added_hh_idx, new_linked = \
        model.transition(hh, iter_var,
                         linked_tables={'linked': (p, 'household_id')})
    new.loc[added_hh_idx, "building_id"] = -1
    orca.add_table("households", new)
    orca.add_table("persons", new_linked['linked'])

@orca.step()
def jobs_transition(jobs, annual_employment_control_totals, iter_var):
    ct_emp = annual_employment_control_totals.to_frame()
    ct_emp = ct_emp.reset_index().set_index('year')
    tran = transition.TabularTotalsTransition(ct_emp, 'total_number_of_jobs')
    model = transition.TransitionModel(tran)
    j = jobs.to_frame(jobs.local_columns)
    new, added_jobs_idx, new_linked = model.transition(j, iter_var)
    new.loc[added_jobs_idx, "building_id"] = -1
    orca.add_table("jobs", new)

@orca.step()
def government_jobs_scaling_model(jobs):
    jobs = jobs.to_frame(jobs.local_columns)
    government_sectors = [18, 19, 20]
    def random_choice(chooser_ids, alternative_ids, probabilities):
        choices = pd.Series([np.nan] * len(chooser_ids), index=chooser_ids)
        chosen = np.random.choice(
            alternative_ids, size=len(chooser_ids), replace=True, p=probabilities)
        choices[chooser_ids] = chosen
        return choices
    jobs_to_place = jobs[jobs.building_id.isnull().values]
    if len(jobs_to_place)>0:
        segments = jobs_to_place.groupby(['large_area_id','sector_id'])
        for name,segment in segments:
            large_area_id = int(name[0])
            sector = int(name[1])
            if sector in government_sectors:
                jobs_to_place = segment.index.values
                counts_by_bid = jobs[(jobs.sector_id == sector).values*(jobs.large_area_id==large_area_id)].groupby(['building_id']).size()
                prop_by_bid = counts_by_bid/counts_by_bid.sum()
                choices = random_choice(jobs_to_place, prop_by_bid.index.values, prop_by_bid.values)
                jobs.loc[choices.index, 'building_id'] = choices.values
        orca.add_table("jobs", jobs)
        
@orca.step()
def refiner(jobs, households, buildings, iter_var):
    jobs = jobs.to_frame()
    households = households.to_frame()
    refinements1 = pd.read_csv("data/refinements.csv")
    refinements2 = pd.read_csv("data/employment_events.csv")
    refinements = pd.concat([refinements1,refinements2])
    refinements = refinements[refinements.year == iter_var]
    if len(refinements) > 0:
        def relocate_agents(agents, agent_type, filter_expression, location_type, location_id, number_of_agents):
            agents = agents.query(filter_expression)
            if location_type == 'zone':
                new_building_id = buildings.zone_id[buildings.zone_id == location_id].index.values[0]
                agent_pool = agents[agents.zone_id != location_id]
            if location_type == 'parcel':
                try:
                    new_building_id = buildings.parcel_id[buildings.parcel_id == location_id].index.values[0]
                except:
                    print 'No building in %s %s.' % (location_type,location_id)
                    return
                agent_pool = agents[agents.parcel_id != location_id]
            shuffled_ids = agent_pool.index.values
            np.random.shuffle(shuffled_ids)
            agents_to_relocate = shuffled_ids[:number_of_agents]
            if agent_type == 'households':
                idx_agents_to_relocate = np.in1d(households.index.values, agents_to_relocate)
                households.building_id[idx_agents_to_relocate] = new_building_id
            if agent_type == 'jobs':
                idx_agents_to_relocate = np.in1d(jobs.index.values, agents_to_relocate)
                jobs.building_id[idx_agents_to_relocate] = new_building_id

        def unplace_agents(agents, agent_type, filter_expression, location_type, location_id, number_of_agents):
            agents = agents.query(filter_expression)
            if location_type == 'zone':
                agent_pool = agents[agents.zone_id == location_id]
            if location_type == 'parcel':
                agent_pool = agents[agents.parcel_id == location_id]
            if len(agent_pool) >= number_of_agents:
                shuffled_ids = agent_pool.index.values
                np.random.shuffle(shuffled_ids)
                agents_to_relocate = shuffled_ids[:number_of_agents]
                if agent_type == 'households':
                    idx_agents_to_relocate = np.in1d(households.index.values, agents_to_relocate)
                    households.building_id[idx_agents_to_relocate] = -1
                if agent_type == 'jobs':
                    idx_agents_to_relocate = np.in1d(jobs.index.values, agents_to_relocate)
                    jobs.building_id[idx_agents_to_relocate] = -1
        for idx in refinements.index.values:
            record = refinements[refinements.index.values == idx]
            action = record.action.values[0]
            agent_dataset = record.agent_dataset.values[0]
            filter_expression = record.filter_expression.values[0]
            amount = record.amount.values[0]
            location_id = record.location_id.values[0]
            location_type = record.location_type.values[0]
            if action == 'add':
                if agent_dataset == 'job':
                    relocate_agents(jobs, 'jobs', filter_expression, location_type, location_id, amount)
                if agent_dataset == 'household':
                    relocate_agents(households, 'households', filter_expression, location_type, location_id, amount)
            if action in ['delete', 'subtract']:
                if agent_dataset == 'job':
                    unplace_agents(jobs, 'jobs', filter_expression, location_type, location_id, amount)
                if agent_dataset == 'household':
                    unplace_agents(households, 'households', filter_expression, location_type, location_id, amount)
                    
        orca.add_table('jobs',jobs)
        orca.add_table('households',households)

@orca.step()
def scheduled_development_events(buildings, iter_var):
    sched_dev = pd.read_csv("data/scheduled_development_events.csv")
    sched_dev[sched_dev.year_built==iter_var]
    sched_dev['building_sqft'] = 0
    sched_dev["sqft_price_res"] = 0
    sched_dev["sqft_price_nonres"] = 0
    if len(sched_dev) > 0:
        max_bid = buildings.index.values.max()
        idx = np.arange(max_bid + 1,max_bid+len(sched_dev)+1)
        sched_dev['building_id'] = idx
        sched_dev = sched_dev.set_index('building_id')
        from urbansim.developer.developer import Developer
        merge = Developer(pd.DataFrame({})).merge
        b = buildings.to_frame(buildings.local_columns)
        all_buildings = merge(b,sched_dev[b.columns])
        orca.add_table("buildings", all_buildings)
    
@orca.step()
def price_vars(net):
    nodes = networks.from_yaml(net, "networks.yaml")
    print nodes.describe()
    print pd.Series(nodes.index).describe()
    orca.add_table("nodes_prices", nodes)
    
@orca.step()
def feasibility(parcels):
    pfc = sqftproforma.SqFtProFormaConfig()
    # Adjust cost downwards based on RS Means test factor
    pfc.costs = {btype: list(np.array(pfc.costs[btype])*.5) for btype in pfc.costs}
    # Adjust price downwards based on RS Means test factor
    pfc.parking_cost_d = {ptype: pfc.parking_cost_d[ptype]*.5 for ptype in pfc.parking_cost_d}
    
    utils.run_feasibility(parcels,
                          variables.parcel_average_price,
                          variables.parcel_is_allowed,
                          to_yearly=True, config=pfc)
            
def random_type(form):
    form_to_btype = orca.get_injectable("form_to_btype")
    return random.choice(form_to_btype[form])

def add_extra_columns_res(df):
    for col in ['building_id_old','improvement_value','land_area','tax_exempt','sqft_price_nonres','sqft_price_res']:
        df[col] = 0
    df['sqft_per_unit'] = 1500
    df = df.fillna(0)
    return df
    
def add_extra_columns_nonres(df):
    for col in ['building_id_old','improvement_value','land_area','tax_exempt','sqft_price_nonres','sqft_price_res']:
        df[col] = 0
    df['sqft_per_unit'] = 0
    df = df.fillna(0)
    return df

@orca.step()
def residential_developer(feasibility, households, buildings, parcels, iter_var):
    utils.run_developer("residential",
                        households,
                        buildings,
                        "residential_units",
                        parcels.parcel_size,
                        parcels.ave_unit_size,
                        parcels.total_units,
                        feasibility,
                        year=iter_var,
                        target_vacancy=.20,
                        form_to_btype_callback=random_type,
                        add_more_columns_callback=add_extra_columns_res,
                        bldg_sqft_per_job=400.0)


@orca.step()
def non_residential_developer(feasibility, jobs, buildings, parcels, iter_var):
    utils.run_developer(["office", "retail", "industrial"],
                        jobs,
                        buildings,
                        "job_spaces",
                        parcels.parcel_size,
                        parcels.ave_unit_size,
                        parcels.total_job_spaces,
                        feasibility,
                        year=iter_var,
                        target_vacancy=.60,
                        form_to_btype_callback=random_type,
                        add_more_columns_callback=add_extra_columns_nonres,
                        residential=False,
                        bldg_sqft_per_job=400.0)
            
@orca.step()
def build_networks(parcels):
##  'mgf14_walk': {'cost1': 'meters',
##                  'edges': 'edges_mgf14_walk',
##                  'nodes': 'nodes_mgf14_walk'}
    network='mgf14_walk'
    st = pd.HDFStore(os.path.join(misc.data_dir(), "semcog_networks.h5"), "r")
    nodes, edges = st['nodes_'+network], st['edges_'+network]
    net = pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"],
                       edges[["feet"]])
    net.precompute(2000)
    orca.add_injectable("net", net)
    
    p = parcels.to_frame()
    p['x'] = p.centroid_x
    p['y'] = p.centroid_y
    p['_node_id'] = net.get_node_ids(p['x'], p['y'])
    orca.add_table("parcels", p)
    
    
@orca.step()
def neighborhood_vars(net, jobs, households, buildings):
    b = buildings.to_frame(['large_area_id'])
    j = jobs.to_frame(jobs.local_columns)
    h = households.to_frame(households.local_columns)
    idx_invalid_building_id = np.in1d(j.building_id,b.index.values)==False
    if idx_invalid_building_id.sum() > 0:
        j.building_id[idx_invalid_building_id] = np.random.choice(b[np.in1d(b.large_area_id,j[idx_invalid_building_id].large_area_id)].index.values,idx_invalid_building_id.sum())
        orca.add_table("jobs", j)
    idx_invalid_building_id = np.in1d(h.building_id,b.index.values)==False
    if idx_invalid_building_id.sum() > 0:
        h.building_id[idx_invalid_building_id] = np.random.choice(b[np.in1d(b.large_area_id,h[idx_invalid_building_id].large_area_id)].index.values,idx_invalid_building_id.sum())
        orca.add_table("households", h)
    nodes = networks.from_yaml(net, "networks.yaml")
    print nodes.describe()
    print pd.Series(nodes.index).describe()
    orca.add_table("nodes", nodes)
    
@orca.step('gq_model') #group quarters
def gq_model(iter_var, tazcounts2015gq, tazcounts2020gq, tazcounts2025gq, tazcounts2030gq, tazcounts2035gq, tazcounts2040gq):
    # Configuration
    max_iterations = 500
    convergence_criteria = .000001
    first_year_to_run = 2015

    # Function used to allocate aggregate GQ from large area controls to TAZ (to form the row marginal)
    def random_choice(chooser_ids, alternative_ids, probabilities):
        choices = pd.Series([np.nan] * len(chooser_ids), index=chooser_ids)
        chosen = np.random.choice(
            alternative_ids, size=len(chooser_ids), replace=True, p=probabilities)
        choices[chooser_ids] = chosen
        return choices

    tazcounts = pd.read_csv('gq/tazcounts.csv').set_index('tazce10')
    gqcontrols = pd.read_csv('gq/largearea_gq_controls.csv')

    # Determine years to run from the control totals
    years = []
    for col in gqcontrols.columns:
        try: 
            if int(col) >= first_year_to_run:
                years.append(col)
        except:
            pass
            
    if iter_var==np.array([int(yr) for yr in years]).min():
        for year in years:
            print year
            for lid in np.unique(gqcontrols.largearea_id):
                print 'Large area id %s' % lid
                gq_lid = gqcontrols[['age_grp',year]][gqcontrols.largearea_id==lid]
                gq_lid = gq_lid.set_index('age_grp')
                tazcounts_lid = tazcounts[['gq04','gq517','gq1834','gq3564','gq65plus']][tazcounts.largearea_id==lid]
                taz_sum = tazcounts_lid.gq04 + tazcounts_lid.gq517 + tazcounts_lid.gq1834 + tazcounts_lid.gq3564 + tazcounts_lid.gq65plus
                diff = gq_lid.sum().values[0] - taz_sum.sum()
                # print 'GQ change is %s' % diff
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
                    ##IPF procedure
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
                            rounded = np.round(seed)
                            rounding_error = marginal2.sum() - rounded.sum().sum()
                            for col in rounded.columns:
                                tazcounts.loc[rounded.index, col] = rounded[col].values
                            break
            tazcounts.to_csv('gq/tazcounts%s.csv'%year)
            orca.add_table('tazcounts%sgq'%year,tazcounts.copy())
            
@orca.step()
def travel_model(iter_var, travel_data, buildings, parcels, households, persons, jobs):
    if iter_var in [2015, 2020, 2025, 2030, 2035, 2040]:
        datatable = 'TAZ Data Table'
        joinfield = 'ZoneID'
        
        input_dir = './/runs//'  ##Where TM expects input
        input_file = input_dir + 'tm_input.tab'
        output_dir = './/data//'  ##Where TM outputs
        output_file = 'tm_output.txt'
        
        def delete_dcc_file(dcc_file):
            if os.path.exists(dcc_file):
                os.remove(dcc_file)
                
        delete_dcc_file(os.path.splitext(input_file)[0] + '.dcc' )
        
        parcels = parcels.to_frame()#(['zone_id','parcel_sqft'])
        hh = households.to_frame()
        persons = persons.to_frame()
        jobs = jobs.to_frame()

        zonal_indicators = pd.DataFrame(index=np.unique(parcels.zone_id.values))
        zonal_indicators['AcresTotal'] = parcels.groupby('zone_id').parcel_sqft.sum()/43560.0
        zonal_indicators['Households'] = hh.groupby('zone_id').size()
        zonal_indicators['HHPop'] = hh.groupby('zone_id').persons.sum()
        zonal_indicators['EmpPrinc'] = jobs.groupby('zone_id').size()
        #zonal_indicators['workers'] = hh.groupby('zone_id').workers.sum()
        zonal_indicators['Agegrp1'] = persons[persons.age<=4].groupby('zone_id').size() #???
        zonal_indicators['Agegrp2'] = persons[(persons.age>=5)*(persons.age<=17)].groupby('zone_id').size() #???
        zonal_indicators['Agegrp3'] = persons[(persons.age>=18)*(persons.age<=34)].groupby('zone_id').size() #???
        zonal_indicators['Age_18to34'] = persons[(persons.age>=18)*(persons.age<=34)].groupby('zone_id').size()
        zonal_indicators['Agegrp4'] = persons[(persons.age>=35)*(persons.age<=64)].groupby('zone_id').size() #???
        zonal_indicators['Agegrp5'] = persons[persons.age>=65].groupby('zone_id').size() #???
        enroll_ratios = pd.read_csv("data/schdic_taz10.csv")
        school_age_by_district = pd.DataFrame({'children':persons[(persons.age>=5)*(persons.age<=17)].groupby('school_district_id').size()})
        enroll_ratios = pd.merge(enroll_ratios,school_age_by_district,left_on='school_district_id',right_index=True)
        enroll_ratios['enrolled'] = enroll_ratios.enroll_ratio*enroll_ratios.children
        enrolled = enroll_ratios.groupby('zone_id').enrolled.sum()
        zonal_indicators['K12Enroll'] = np.round(enrolled)
        zonal_indicators['PopDens'] = zonal_indicators.HHPop/(parcels.groupby('zone_id').parcel_sqft.sum()/43560)
        zonal_indicators['EmpDens'] = zonal_indicators.EmpPrinc/(parcels.groupby('zone_id').parcel_sqft.sum()/43560)
        # zonal_indicators['EmpBasic'] = jobs[jobs.sector_id.isin([1,3])].groupby('zone_id').size()
        # zonal_indicators['EmpNonBas'] = jobs[~jobs.sector_id.isin([1,3])].groupby('zone_id').size()
        zonal_indicators['Natural_Resource_and_Mining'] = jobs[jobs.sector_id==1].groupby('zone_id').size()
        #zonal_indicators['sector2'] = jobs[jobs.sector_id==2].groupby('zone_id').size()
        zonal_indicators['Manufacturing'] = jobs[jobs.sector_id==3].groupby('zone_id').size()
        zonal_indicators['Wholesale_Trade'] = jobs[jobs.sector_id==4].groupby('zone_id').size()
        zonal_indicators['Retail_Trade'] = jobs[jobs.sector_id==5].groupby('zone_id').size()
        zonal_indicators['Transportation_and_Warehousing'] = jobs[jobs.sector_id==6].groupby('zone_id').size()
        zonal_indicators['Utilities'] = jobs[jobs.sector_id==7].groupby('zone_id').size()
        zonal_indicators['Information'] = jobs[jobs.sector_id==8].groupby('zone_id').size()
        zonal_indicators['Financial_Service'] = jobs[jobs.sector_id==9].groupby('zone_id').size()
        zonal_indicators['Professional_Science_Tec'] = jobs[jobs.sector_id==10].groupby('zone_id').size()
        zonal_indicators['Management_of_CompEnt'] = jobs[jobs.sector_id==11].groupby('zone_id').size()
        zonal_indicators['Administrative_Support_and_WM'] = jobs[jobs.sector_id==12].groupby('zone_id').size()
        zonal_indicators['Education_Services'] = jobs[jobs.sector_id==13].groupby('zone_id').size()
        # zonal_indicators['sector14'] = jobs[jobs.sector_id==14].groupby('zone_id').size()
        # zonal_indicators['sector15'] = jobs[jobs.sector_id==15].groupby('zone_id').size()
        zonal_indicators['Health_Care_and_SocialSer'] = jobs[np.in1d(jobs.sector_id,[14,15,19])].groupby('zone_id').size()
        zonal_indicators['Leisure_and_Hospitality'] = jobs[jobs.sector_id==16].groupby('zone_id').size()
        zonal_indicators['Other_Services'] = jobs[jobs.sector_id==17].groupby('zone_id').size()
        zonal_indicators['sector18'] = jobs[jobs.sector_id==18].groupby('zone_id').size()
        zonal_indicators['sector19'] = jobs[jobs.sector_id==19].groupby('zone_id').size()
        zonal_indicators['Public_Administration'] = jobs[jobs.sector_id==20].groupby('zone_id').size()
        
        hh['schoolkids'] = persons[(persons.age>=5)*(persons.age<=17)].groupby('household_id').size()
        hh.schoolkids = hh.schoolkids.fillna(0)
        zonal_indicators['PrCh21'] = hh[(hh.persons==2)*(hh.schoolkids==1)].groupby('zone_id').size()
        zonal_indicators['PrCh31'] = hh[(hh.persons==3)*(hh.schoolkids==1)].groupby('zone_id').size()
        zonal_indicators['PrCh32'] = hh[(hh.persons==3)*(hh.schoolkids==2)].groupby('zone_id').size()
        zonal_indicators['PrCh41'] = hh[(hh.persons==4)*(hh.schoolkids==1)].groupby('zone_id').size()
        zonal_indicators['PrCh42'] = hh[(hh.persons==4)*(hh.schoolkids==2)].groupby('zone_id').size()
        zonal_indicators['PrCh43'] = hh[(hh.persons==4)*(hh.schoolkids>=3)].groupby('zone_id').size()
        zonal_indicators['PrCh51'] = hh[(hh.persons==5)*(hh.schoolkids==1)].groupby('zone_id').size()
        zonal_indicators['PrCh52'] = hh[(hh.persons==5)*(hh.schoolkids==2)].groupby('zone_id').size()
        zonal_indicators['PrCh53'] = hh[(hh.persons==5)*(hh.schoolkids>=3)].groupby('zone_id').size()
        hh['quartile'] = pd.Series(pd.qcut(hh.income,4,labels=False), index=hh.index)+1
        zonal_indicators['Inc1HHsze1'] = hh[(hh.persons==1)*(hh.quartile==1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze1'] = hh[(hh.persons==1)*(hh.quartile==2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze1'] = hh[(hh.persons==1)*(hh.quartile==3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze1'] = hh[(hh.persons==1)*(hh.quartile==4)].groupby('zone_id').size()
        zonal_indicators['Inc1HHsze2'] = hh[(hh.persons==2)*(hh.quartile==1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze2'] = hh[(hh.persons==2)*(hh.quartile==2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze2'] = hh[(hh.persons==2)*(hh.quartile==3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze2'] = hh[(hh.persons==2)*(hh.quartile==4)].groupby('zone_id').size()
        zonal_indicators['Inc1HHsze3'] = hh[(hh.persons==3)*(hh.quartile==1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze3'] = hh[(hh.persons==3)*(hh.quartile==2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze3'] = hh[(hh.persons==3)*(hh.quartile==3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze3'] = hh[(hh.persons==3)*(hh.quartile==4)].groupby('zone_id').size()
        zonal_indicators['Inc1HHsze4'] = hh[(hh.persons==4)*(hh.quartile==1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze4'] = hh[(hh.persons==4)*(hh.quartile==2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze4'] = hh[(hh.persons==4)*(hh.quartile==3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze4'] = hh[(hh.persons==4)*(hh.quartile==4)].groupby('zone_id').size()
        zonal_indicators['Inc1HHsze5p'] = hh[(hh.persons==5)*(hh.quartile==1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze5p'] = hh[(hh.persons==5)*(hh.quartile==2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze5p'] = hh[(hh.persons==5)*(hh.quartile==3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze5p'] = hh[(hh.persons==5)*(hh.quartile==4)].groupby('zone_id').size()
        zonal_indicators['WkAu10'] = hh[(hh.workers==1)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['WkAu11'] = hh[(hh.workers==1)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['WkAu12'] = hh[(hh.workers==1)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['WkAu13'] = hh[(hh.workers==1)*(hh.cars>=3)].groupby('zone_id').size()
        zonal_indicators['WkAu20'] = hh[(hh.workers==2)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['WkAu21'] = hh[(hh.workers==2)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['WkAu22'] = hh[(hh.workers==2)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['WkAu23'] = hh[(hh.workers==2)*(hh.cars>=3)].groupby('zone_id').size()
        zonal_indicators['WkAu30'] = hh[(hh.workers>=3)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['WkAu31'] = hh[(hh.workers>=3)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['WkAu32'] = hh[(hh.workers>=3)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['WkAu33'] = hh[(hh.workers>=3)*(hh.cars>=3)].groupby('zone_id').size()

        zonal_indicators['PrAu10'] = hh[(hh.persons==1)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['PrAu11'] = hh[(hh.persons==1)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['PrAu12'] = hh[(hh.persons==1)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['PrAu13'] = hh[(hh.persons==1)*(hh.cars>=3)].groupby('zone_id').size()
        zonal_indicators['PrAu20'] = hh[(hh.persons==2)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['PrAu21'] = hh[(hh.persons==2)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['PrAu22'] = hh[(hh.persons==2)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['PrAu23'] = hh[(hh.persons==2)*(hh.cars>=3)].groupby('zone_id').size()
        zonal_indicators['PrAu30'] = hh[(hh.persons==3)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['PrAu31'] = hh[(hh.persons==3)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['PrAu32'] = hh[(hh.persons==3)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['PrAu33'] = hh[(hh.persons==3)*(hh.cars>=3)].groupby('zone_id').size()
        zonal_indicators['PrAu40'] = hh[(hh.persons==4)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['PrAu41'] = hh[(hh.persons==4)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['PrAu42'] = hh[(hh.persons==4)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['PrAu43'] = hh[(hh.persons==4)*(hh.cars>=3)].groupby('zone_id').size()
        zonal_indicators['PrAu50'] = hh[(hh.persons==5)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['PrAu51'] = hh[(hh.persons==5)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['PrAu52'] = hh[(hh.persons==5)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['PrAu53'] = hh[(hh.persons==5)*(hh.cars>=3)].groupby('zone_id').size()
		
        # zonal_indicators['inc1nc'] = hh[(hh.quartile==1)*(hh.children==0)].groupby('zone_id').size()
        # zonal_indicators['inc1wc'] = hh[(hh.quartile==1)*(hh.children>0)].groupby('zone_id').size()
        # zonal_indicators['inc2nc'] = hh[(hh.quartile==2)*(hh.children==0)].groupby('zone_id').size()
        # zonal_indicators['inc2wc'] = hh[(hh.quartile==2)*(hh.children>0)].groupby('zone_id').size()
        # zonal_indicators['inc3nc'] = hh[(hh.quartile==3)*(hh.children==0)].groupby('zone_id').size()
        # zonal_indicators['inc3wc'] = hh[(hh.quartile==3)*(hh.children>0)].groupby('zone_id').size()
        # zonal_indicators['inc4nc'] = hh[(hh.quartile==4)*(hh.children==0)].groupby('zone_id').size()
        # zonal_indicators['inc4wc'] = hh[(hh.quartile==4)*(hh.children>0)].groupby('zone_id').size()

        zonal_indicators['Inc1w0'] = hh[(hh.quartile==1)*(hh.workers==0)].groupby('zone_id').size()
        zonal_indicators['Inc1w1'] = hh[(hh.quartile==1)*(hh.workers==1)].groupby('zone_id').size()
        zonal_indicators['Inc1w2'] = hh[(hh.quartile==1)*(hh.workers==2)].groupby('zone_id').size()
        zonal_indicators['Inc1w3p'] = hh[(hh.quartile==1)*(hh.workers>=3)].groupby('zone_id').size()
        zonal_indicators['Inc2w0'] = hh[(hh.quartile==2)*(hh.workers==0)].groupby('zone_id').size()
        zonal_indicators['Inc2w1'] = hh[(hh.quartile==2)*(hh.workers==1)].groupby('zone_id').size()
        zonal_indicators['Inc2w2'] = hh[(hh.quartile==2)*(hh.workers==2)].groupby('zone_id').size()
        zonal_indicators['Inc2w3p'] = hh[(hh.quartile==2)*(hh.workers>=3)].groupby('zone_id').size()
        zonal_indicators['Inc3w0'] = hh[(hh.quartile==3)*(hh.workers==0)].groupby('zone_id').size()
        zonal_indicators['Inc3w1'] = hh[(hh.quartile==3)*(hh.workers==1)].groupby('zone_id').size()
        zonal_indicators['Inc3w2'] = hh[(hh.quartile==3)*(hh.workers==2)].groupby('zone_id').size()
        zonal_indicators['Inc3w3p'] = hh[(hh.quartile==3)*(hh.workers>=3)].groupby('zone_id').size()
        zonal_indicators['Inc4w0'] = hh[(hh.quartile==4)*(hh.workers==0)].groupby('zone_id').size()
        zonal_indicators['Inc4w1'] = hh[(hh.quartile==4)*(hh.workers==1)].groupby('zone_id').size()
        zonal_indicators['Inc4w2'] = hh[(hh.quartile==4)*(hh.workers==2)].groupby('zone_id').size()
        zonal_indicators['Inc4w3p'] = hh[(hh.quartile==4)*(hh.workers>=3)].groupby('zone_id').size()
        
        zonal_indicators['Workers4HH_IncomeGroup1'] = hh[hh.quartile==1].groupby('zone_id').workers.sum()
        zonal_indicators['Workers4HH_IncomeGroup2'] = hh[hh.quartile==2].groupby('zone_id').workers.sum()
        zonal_indicators['Workers4HH_IncomeGroup3'] = hh[hh.quartile==3].groupby('zone_id').workers.sum()
        zonal_indicators['Workers4HH_IncomeGroup4'] = hh[hh.quartile==4].groupby('zone_id').workers.sum()
        
        if os.path.exists('gq/tazcounts%s.csv'%iter_var):
            gq = pd.read_csv('gq/tazcounts%s.csv'%iter_var).set_index('tazce10')
            gq['GrPop'] = gq.gq04+gq.gq517+gq.gq1834+gq.gq3564+gq.gq65plus
            zonal_indicators['GrPop'] = gq['GrPop']
            zonal_indicators['Population'] = zonal_indicators['GrPop'] + zonal_indicators['HHPop']
            
        ##Update parcel land_use_type_id
        buildings = buildings.to_frame(['parcel_id','building_type_id','year_built'])
        new_construction = buildings[buildings.year_built==iter_var].groupby('parcel_id').building_type_id.median()
        if len(new_construction) > 0:
            parcels.loc[new_construction.index, 'land_use_type_id'] = new_construction.values
            orca.add_table("parcels", parcels)
        
        emp_btypes = orca.get_injectable('emp_btypes')
        emp_parcels = buildings[np.in1d(buildings.building_type_id,emp_btypes)].groupby('parcel_id').size().index.values
        parcels['emp'] = 0
        parcels.emp[np.in1d(parcels.index.values,emp_parcels)] = 1
        parcels['emp_acreage'] = parcels.emp*parcels.parcel_sqft/43560.0
        zonal_indicators['AcresEmp'] = parcels.groupby('zone_id').emp_acreage.sum()
        
        zonal_indicators['TAZCE10_N'] = zonal_indicators.index.values
        #zonal_indicators = zonal_indicators.fillna(0).reset_index().rename({'ZoneID':'TAZCE10_N'})
        
        taz_table = pd.read_csv("data/taz_table.csv")
        
        merged = pd.merge(taz_table, zonal_indicators, left_on='TAZCE10_N', right_on='TAZCE10_N', how='left')
        
        merged.to_csv(input_file, sep='\t', index = False)
        
        #######################################################################
        ####    TRANSCAD INTERACTIONS #########################################
        #######################################################################
        if orca.get_injectable("transcad_available") == True:
            transcad.transcad_interaction(merged, taz_table)
        
        
def _print_number_unplaced(df, fieldname="building_id"):
    """
    Just an internal function to use to compute and print info on the number
    of unplaced agents.
    """
    counts = (df[fieldname]==-1).sum()
    print "Total currently unplaced: %d" % counts

import numpy as np
import pandas as pd


def calculate(dset):
    def merge_buildings_parcels(buildings, parcels):
        return pd.merge(buildings, parcels, left_on='parcel_id', right_index=True)

    def unit_price_res_column(buildings):
        """
        Calculate residential unit price as improvement_value per residential unit.
        
        """
        buildings['unit_price_res'] = buildings.improvement_value / buildings.residential_units
        buildings['unit_price_res'][buildings['residential_units'] == 0] = 0
        return buildings

    def unit_price_nonres_column(buildings):
        """
        Calculate price per nonres-sqft as improvement_value per non_residential_sqft.
        
        """
        buildings['unit_price_nonres'] = buildings.improvement_value / buildings.non_residential_sqft
        buildings['unit_price_nonres'][buildings['non_residential_sqft'] == 0] = 0
        return buildings

    def population_density(buildings, households):
        """
        Calculate population density at the zonal level as people per acre
        then broadcast out to the building level.
        
        """
        sqft_per_acre = 43560
        
        bldg_pop = households.groupby('building_id').persons.sum()
        zone_pop = bldg_pop.groupby(buildings.zone_id).sum()
        zone_acres = buildings.parcel_sqft.groupby(buildings.zone_id).sum() / sqft_per_acre
        pop_density = (zone_pop / zone_acres).fillna(0)
        buildings['popden'] = pd.Series(pop_density[buildings.zone_id].values, 
                                        index=buildings.index)
        return buildings

    def crime_rate(buildings, cities):
        """
        Broadcast crime rate from the cities table to buildings.
        
        """
        buildings['crime08'] = cities.crime08[buildings.city_id].values
        return buildings

    def jobs_within_30_min(buildings, travel_data, jobs):
        """
        Calculate the number of jobs within thirty minutes of each building.
        This is actually done at the zonal level and then broadcast
        to buildings.
        
        """
        # The travel_data table has a multi-level index with from_zone_id
        # and to_zone_id. We care about the travel time between zones so
        # we want to move the to_zone_id into the DataFrame as a regular column
        # and then keep all the zone pairs that are less than 30 minutes apart
        zones_within_30_min = (travel_data.reset_index(level='to_zone_id')
                               .query('am_single_vehicle_to_work_travel_time < 30').to_zone_id)

        # The next step is to tabulate the number of jobs in each zone,
        # broadcast that across the zones within range of each other zone,
        # and finally group by the from_zone_id and total all the jobs within range.
        job_counts = jobs.groupby('zone_id').size()
        job_counts = pd.Series(
            job_counts[zones_within_30_min].fillna(0).values, 
            index=zones_within_30_min.index).groupby(level=0).sum()
        buildings['jobs_within_30_min'] = job_counts[buildings.zone_id].fillna(0).values
        return buildings

    buildings = dset.fetch('buildings')
    buildings.non_residential_sqft = buildings.non_residential_sqft.astype('int64')
    parcels = dset.fetch('parcels')
    households = dset.fetch('households')
    jobs = dset.fetch('jobs')

    buildings = merge_buildings_parcels(buildings, parcels)

    buildings = unit_price_res_column(buildings)
    buildings = unit_price_nonres_column(buildings)
    buildings = population_density(buildings, households)
    buildings = crime_rate(buildings, dset.cities)
    buildings = jobs_within_30_min(buildings, dset.travel_data, jobs)

    buildings = pd.merge(buildings, dset.building_sqft_per_job,
                         left_on=['zone_id', 'building_type_id'],
                         right_index=True, how='left')
    buildings['job_spaces'] = buildings.non_residential_sqft/buildings.building_sqft_per_job
    buildings['job_spaces'] = np.round(buildings['job_spaces'].fillna(0)).astype('int')
    buildings.job_spaces[buildings.job_spaces < 0] = 0

    buildings['building_sqft'] = buildings.non_residential_sqft + buildings.sqft_per_unit*buildings.residential_units
    dset.buildings = buildings
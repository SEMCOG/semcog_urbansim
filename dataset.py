import os
import pandas as pd
import numpy as np
from urbansim.utils import dataset, misc
from urbansim.utils.dataset import variable


class SemcogDataset(dataset.Dataset):
    def __init__(self, filename):
        super(SemcogDataset, self).__init__(filename)

    def clear_views(self):
        self.views = {
            "parcels": Parcels(self),
            "households": Households(self),
            "jobs": Jobs(self),
            "buildings": Buildings(self),
            "zones": Zones(self),
            "cities": self.cities
        }

    @staticmethod
    def fetch_nodes():
        # default will fetch off disk unless networks have already been run
        print "WARNING: fetching precomputed nodes off of disk"
        df = pd.read_csv(os.path.join(misc.data_dir(), 'nodes.csv'), index_col='node_id')
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df

    def merge_nodes(self, df):
        return pd.merge(df, self.nodes, left_on="_node_id", right_index=True)


class Parcels(dataset.CustomDataFrame):

    def __init__(self, dset):
        super(Parcels, self).__init__(dset, "parcels")
        self.flds = ["acres", "x", "y"]

    @variable
    def acres(self):
        return "parcels.parcel_sqft / 43560"

    @variable
    def x(self):
        return "parcels.centroid_x"

    @variable
    def y(self):
        return "parcels.centroid_y"


class Households(dataset.CustomDataFrame):

    def __init__(self, dset):
        super(Households, self).__init__(dset, "households")
        self.flds = ["zone_id", "building_id", "income", "x", "y",
                     "persons", "income_quartile", "large_area"]

    @property
    def income_quartile(self):
        return pd.Series(pd.qcut(self.df.income, 4).labels, index=self.df.index)

    @variable
    def zone_id(self):
        return "reindex(buildings.zone_id, households.building_id)"

    @variable
    def x(self):
        return "reindex(buildings.x, households.building_id)"

    @variable
    def y(self):
        return "reindex(buildings.y, households.building_id)"
        
    @variable
    def large_area(self):
        return "reindex(buildings.large_area_id, households.building_id)"


class Jobs(dataset.CustomDataFrame):

    def __init__(self, dset):
        super(Jobs, self).__init__(dset, "jobs")
        self.flds = ["zone_id", "building_id", "home_based_status",
                     "x", "y", "large_area", "sector_id", "parcel_id"]

    @variable
    def zone_id(self):
        return "reindex(buildings.zone_id, jobs.building_id)"
        
    @variable
    def parcel_id(self):
        return "reindex(buildings.parcel_id, jobs.building_id)"

    @variable
    def large_area(self):
        return "reindex(buildings.large_area_id, jobs.building_id)"

    @variable
    def x(self):
        return "reindex(buildings.x, jobs.building_id)"

    @variable
    def y(self):
        return "reindex(buildings.y, jobs.building_id)"


class Zones(dataset.CustomDataFrame):

    def __init__(self, dset):
        super(Zones, self).__init__(dset, "zones")

    @variable
    def popden(self):
        return "households.persons.groupby(households.zone_id).sum() / parcels.acres.groupby(parcels.zone_id).sum()"

    @property
    def jobs_within_30_min(self):
        return misc.compute_range(self.dset.travel_data,
                                  self.dset.jobs.groupby('zone_id').size(),
                                  "am_single_vehicle_to_work_travel_time",
                                  30, agg=np.sum)


class Buildings(dataset.CustomDataFrame):

    def __init__(self, dset):
        self.flds = ["crime08", "popden", "unit_price_res_est", "sqft_per_unit",
                     "unit_price_nonres_est", "building_sqft", "job_spaces",
                     "jobs_within_30_min", "non_residential_sqft",
                     "residential_units", "year_built", "stories",
                     "tax_exempt", "building_type_id", "dist_hwy", "dist_road",
                     "x", "y", "land_area", "zone_id", "large_area_id", "parcel_id",
                     "_node_id"]
        super(Buildings, self).__init__(dset, "buildings")

    @variable
    def _node_id(self):
        return "reindex(parcels._node_id, buildings.parcel_id)"

    @variable
    def x(self):
        return "reindex(parcels.x, buildings.parcel_id)"

    @variable
    def y(self):
        return "reindex(parcels.y, buildings.parcel_id)"

    @variable
    def dist_hwy(self):
        return "reindex(parcels.dist_hwy, buildings.parcel_id)"

    @variable
    def dist_road(self):
        return "reindex(parcels.dist_road, buildings.parcel_id)"

    @variable
    def zone_id(self):
        return "reindex(parcels.zone_id, buildings.parcel_id)"

    @variable
    def city_id(self):
        return "reindex(parcels.city_id, buildings.parcel_id)"
        
    @variable
    def large_area_id(self):
        return "reindex(parcels.large_area_id, buildings.parcel_id)"

    @variable
    def crime08(self):
        return 'reindex(cities.crime08, buildings.city_id)'

    @variable
    def popden(self):
        return "reindex(zones.popden, buildings.zone_id)"

    @variable
    def building_sqft(self):
        return "buildings.non_residential_sqft + buildings.sqft_per_unit*buildings.residential_units"

    @variable
    def unit_price_nonres_est(self):
        return "buildings.improvement_value / buildings.non_residential_sqft"

    @variable
    def unit_price_res_est(self):
        return "buildings.improvement_value / (buildings.sqft_per_unit * buildings.residential_units)"

    @property
    def building_sqft_per_job(self):
        return pd.merge(self.build_df(flds=['zone_id', 'building_type_id']),
                        self.dset.building_sqft_per_job,
                        left_on=['zone_id', 'building_type_id'],
                        right_index=True, how='left').building_sqft_per_job

    @property
    def job_spaces(self):
        job_spaces = self.non_residential_sqft / self.building_sqft_per_job
        job_spaces[np.isinf(job_spaces)] = np.nan
        job_spaces[job_spaces < 0] = 0
        job_spaces = job_spaces.fillna(0).round().astype('int')
        return job_spaces

    @variable
    def jobs_within_30_min(self):
        return "reindex(zones.jobs_within_30_min, buildings.zone_id)"

LocalDataset = SemcogDataset

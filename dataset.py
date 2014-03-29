from urbansim.urbansim.dataset import Dataset


class SemcogDataset(Dataset):
    def fetch_buildings(self):
        # population density = zonal population / acres per zone
        parcels = self.store.get('parcels')

        # zone population
        bldg_pop = (self.store.get('households')
                    .groupby('building_id')
                    .persons.sum())
        parcel_ids = self.store.get('buildings').parcel_id
        parcel_pop = bldg_pop.groupby(parcel_ids).sum()
        zone_pop = parcel_pop.groupby(parcels.zone_id).sum()

        # zone acreage
        zone_acres = parcels.groupby('zone_id').parcel_sqft.sum() / 43560.0

        pop_density = zone_pop / zone_acres

        bldg_zone_id = parcels.zone_id[parcel_ids]
        bldg_zone_id.index = parcel_ids.index

        buildings = self.store.get('buildings')
        buildings['popden'] = pop_density[bldg_zone_id].values

        # crime rate
        crimerate = self.store.get('cities').crime08
        parcel_cr = crimerate[parcels.city_id]
        parcel_cr.index = parcels.index
        buildings['crime08'] = parcel_cr[parcel_ids].values

        # jobs within thirty minutes
        travel_data = self.fetch('travel_data').reset_index(level='to_zone_id')
        travel_data = travel_data[
            travel_data.am_single_vehicle_to_work_travel_time < 30]
        job_counts = travel_data.to_zone_id.groupby(level=0).count()
        buildings['jobs_within_30_min'] = job_counts[bldg_zone_id].values

        return buildings

import pandas as pd
import numpy as np

if __name__ == "__main__":
    tv = pd.read_csv('~/share/U_RDF2050/base_table/target_vacancies.csv')
    hdf_store = pd.HDFStore( "/media/urbansim/RDF2045/data/base_year/all_semcog_data_02-02-18-final-forecast.h5", mode="r")


    tv_mcd = pd.DataFrame(index = list(hdf_store['semmcds'].index) + [2073])
    tv_mcd.index.name = 'semmcd'
    tv_mcd = tv_mcd.reset_index()

    tv_mcd['year'] = [range(2015, 2051) for _ in range(tv_mcd.shape[0])]
    tv_mcd['res_target_vacancy_rate']= [np.random.uniform(0.03, 0.15) for _ in range(tv_mcd.shape[0])]
    tv_mcd['non_res_target_vacancy_rate']= [np.random.uniform(0.15, 0.25) for _ in range(tv_mcd.shape[0])]
    tv_mcd = tv_mcd.explode('year')

    tv_mcd.to_csv('~/share/U_RDF2050/base_table/target_vacancies_mcd.csv', index=False)
    hdf_store.close()
    print('Done')
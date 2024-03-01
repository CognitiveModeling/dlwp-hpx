#! /usr/bin/env python3

import cdsapi

c = cdsapi.Client()

#c.retrieve(
#    'reanalysis-era5-pressure-levels',
#    {
#        'product_type': 'reanalysis',
#        'variable': [
#            'geopotential', 'specific_humidity', 'temperature',
#            'u_component_of_wind', 'v_component_of_wind',
#        ],
#        'pressure_level': [
#            '50', '100', '150',
#            '200', '250', '300',
#            '400', '500', '600',
#            '700', '850', '925',
#            '1000',
#        ],
#        'year': '2018',
#        'month': [
#            '01', '02', '03',
#            '04', '05', '06',
#            '07', '08', '09',
#            '10', '11', '12',
#        ],
#        'day': [
#            '01', '02', '03',
#            '04', '05', '06',
#            '07', '08', '09',
#            '10', '11', '12',
#            '13', '14', '15',
#            '16', '17', '18',
#            '19', '20', '21',
#            '22', '23', '24',
#            '25', '26', '27',
#            '28', '29', '30',
#            '31',
#        ],
#        'time': '00:00',
#        'format': 'grib',
#    },
#    'pressure_levels_2018_24h.grib')

#c.retrieve(
#    'reanalysis-era5-single-levels',
#    {
#        'product_type': 'reanalysis',
#        'format': 'netcdf',
#        'variable': [
#            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
#            'mean_sea_level_pressure',
#        ],
#        'year': '2018',
#        'month': [
#            '01', '02', '03',
#            '04', '05', '06',
#            '07', '08', '09',
#            '10', '11', '12',
#        ],
#        'day': [
#            '01', '02', '03',
#            '04', '05', '06',
#            '07', '08', '09',
#            '10', '11', '12',
#            '13', '14', '15',
#            '16', '17', '18',
#            '19', '20', '21',
#            '22', '23', '24',
#            '25', '26', '27',
#            '28', '29', '30',
#            '31',
#        ],
#        'time': '00:00',
#    },
#    'single_levels_2018_24h.nc')

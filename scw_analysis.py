import geopandas as gpd
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import glob2
import rasterio
from rasterstats import zonal_stats
import os
import json
import urllib.parse
import requests
import time

def query_arcgis_feature_server(url_feature_server=''):
    '''
    This function downloads all of the features available on a given ArcGIS
    feature server. The function is written to bypass the limitations imposed
    by the online service, such as only returning up to 1,000 or 2,000 featues
    at a time.

    Parameters
    ----------
    url_feature_server : string
        Sting containing the URL of the service API you want to query. It should
        end in a forward slash and look something like this:
        'https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Counties/FeatureServer/0/'

    Returns
    -------
    geodata_final : gpd.GeoDataFrame
        This is a GeoDataFrame that contains all of the features from the
        Feature Server. After calling this function, the `geodata_final` object
        can be used to store the data on disk in several different formats
        including, but not limited to, Shapefile (.shp), GeoJSON (.geojson),
        GeoPackage (.gpkg), or PostGIS.
        See https://geopandas.org/en/stable/docs/user_guide/io.html#writing-spatial-data
        for more details.

    '''
    if url_feature_server == '':
        geodata_final = gpd.GeoDataFrame()
        return geodata_final

    # Fixing last character in case the URL provided didn't end in a
    # forward slash
    if url_feature_server[-1] != '/':
        url_feature_server = url_feature_server + '/'

    # Getting the layer definitions. This contains important info such as the
    # name of the column used as feature_ids/object_ids, among other things.
    layer_def = requests.get(url_feature_server + '?f=pjson').json()

    # The `objectIdField` is the column name used for the
    # feature_ids/object_ids
    fid_colname = 'OBJECTID'

    # The `maxRecordCount` tells us the maximum number of records this REST
    # API service can return at once. The code below is written such that we
    # perform multiple calls to the API, each one being short enough never to
    # go beyond this limit.
    record_count_max = layer_def['maxRecordCount']

    # Part of the URL that specifically requests only the object IDs
    url_query_get_ids = (f'query?f=geojson&returnIdsOnly=true'
                         f'&where={fid_colname}+is+not+null')

    url_comb = url_feature_server + url_query_get_ids

    # Getting all the object IDs
    service_request = requests.get(url_comb)
    all_objectids = np.sort(service_request.json()['objectIds'])

    # This variable will store all the parts of the multiple queries. These
    # parts will, at the end, be concatenated into one large GeoDataFrame.
    geodata_parts = []

    # This part of the query is fixed and never actually changes
    url_query_fixed = ('query?f=geojson&outFields=*&where=')

    # Identifying the largest query size allowed per request. This will dictate
    # how many queries will need to be made. We start the search at
    # the max record count, but that generates errors sometimes - the query
    # might time out because it's too big. If the test query times out, we try
    # shrink the query size until the test query goes through without
    # generating a time-out error.
    block_size = min(record_count_max, len(all_objectids))
    worked = False
    while not worked:
        # Moving the "cursors" to their appropriate locations
        id_start = all_objectids[0]
        id_end = all_objectids[block_size-1]

        readable_query_string = (f'{fid_colname}>={id_start} '
                                 f'and {fid_colname}<={id_end}')

        url_query_variable =  urllib.parse.quote(readable_query_string)

        url_comb = url_feature_server + url_query_fixed + url_query_variable

        url_get = requests.get(url_comb)

        if 'error' in url_get.json():
            block_size = int(block_size/2)+1
        else:
            geodata_part = gpd.read_file(url_get.text)

            geodata_parts.append(geodata_part.copy())
            worked = True

    # Performing the actual query to the API multiple times. This skips the
    # first few rows/features in the data because those rows were already
    # captured in the query performed in the code chunk above.
    for i in range(block_size, len(all_objectids), block_size):
        if (i/len(all_objectids) * 100) < 80:
            continue

        print(i/len(all_objectids) * 100)

        # Moving the "cursors" to their appropriate locations and finding the
        # limits of each block
        sub_list = all_objectids[i:i + block_size]
        id_start = sub_list[0]
        id_end = sub_list[-1]

        readable_query_string = (f'{fid_colname}>={id_start} '
                                 f'and {fid_colname}<={id_end}')

        # Encoding from readable text to URL
        url_query_variable =  urllib.parse.quote(readable_query_string)

        # Constructing the full request URL
        url_comb = url_feature_server + url_query_fixed + url_query_variable

        # Actually performing the query and storing its results in a
        # GeoDataFrame
        geodata_part =  (gpd.read_file(url_comb,
                                       driver='GeoJSON'))

        # Appending the result to `geodata_parts`
        if geodata_part.shape[0] > 0:
            geodata_parts.append(geodata_part)

    # Concatenating all of the query parts into one large GeoDataFrame
    geodata_final = (pd.concat(geodata_parts,
                               ignore_index=True)
                     .sort_values(by=fid_colname)
                     .reset_index(drop=True))

    # Checking if any object ID is missing
    ids_queried = set(geodata_final[fid_colname])
    for i,this_id in enumerate(all_objectids):
        if this_id not in ids_queried:
            print('WARNING! The following ObjectID is missing from the final '
                  f'GeoDataFrame: ObjectID={this_id}')
            pass

    # Checking if any object ID is included twice
    geodata_temp = geodata_final[[fid_colname]].copy()
    geodata_temp['temp'] = 1
    geodata_temp = (geodata_temp
                    .groupby(fid_colname)
                    .agg({'temp':'sum'})
                    .reset_index())
    geodata_temp = geodata_temp.loc[geodata_temp['temp']>1].copy()
    for i,this_id in enumerate(geodata_temp[fid_colname].values):
        n_times = geodata_temp['temp'].values[i]
        print('WARNING! The following ObjectID is included multiple times in'
              f'the final GeoDataFrame: ObjectID={this_id}\tOccurrences={n_times}')

    return geodata_final


def download_shp():
    '''
    This function downloads the SCW parcels dataset and saves it as a shapefile.
    '''

    # SCW parcels MapServer url
    url = "https://dpw.gis.lacounty.gov/dpw/rest/services/Safe_Clean_Water_Tax/MapServer/0/"
    # call function to query MapServer
    scw_gdf = query_arcgis_feature_server(url)
    # reproject to projection with feet
    scw_gdf_reproject = scw_gdf.to_crs(2229)
    # save geodataframe as shapefile
    scw_gdf_reproject.to_file('SCW_parcels.shp', engine='pyogrio')

    return


def combine_landcover():
    '''
    This function combines the NOAA landcover dataset with the LA County building outlines dataset. Any pixels that are marked as buildings in the LA County dataset are marked as impervious in the NOAA dataset and then the raster is saved.
    '''

    # NOAA filepath
    noaa_imp = 'ca_2021_ccap_v2_hires_impervious_20240108_reproject.tif'
    # output filepath
    outfile = 'ca_2021_ccap_v2_hires_impervious_20240108_reproject_buildings.tif'
    # buildings filepath
    buildings_file = './LACounty_buildings/LACounty_buildings_reproject.tif'

    # open the NOAA and buildings rasters
    with rasterio.open(noaa_imp) as src_noaa, rasterio.open(buildings_file) as src_buildings:
        profile = src_noaa.profile
        noaa = src_noaa.read(1)
        buildings = src_buildings.read(1)

        # convert any nonimpervous NOAA pixels to impervous if there is a building
        noaa = np.where(buildings == 1, 1, noaa)

    # save the raster
    with rasterio.open(outfile, 'w', **profile) as dst:
        dst.write(noaa,1)

    return


def comparison():
    '''
    This function reverse engineers the SCW tax using the original impervous surface area. The new SCW tax using just the NOAA cover and the NOAA and buildings impervious cover is also calculated.
    '''

    # read in SCW parcels data
    scw_parcels = gpd.read_file('SCW_parcels_noaa_buildings.shp', engine='pyogrio')

    #remove any exempt parcels
    scw_parcels = scw_parcels[scw_parcels['ExmpAny'] == 0]
    #replace original impervious surface area with the appealed impervious surface area if there is one
    scw_parcels.loc[scw_parcels['AppealImpS'].notna(), 'SCW_IMPERM'] = scw_parcels['AppealImpS']
    #divide the surface area by the overlappping parcels which must be done for condos
    scw_parcels['SCW_IMPERM'] = scw_parcels['SCW_IMPERM'] / scw_parcels['SCW_NUM_OC']
    #for any modifying percentages with NULL, replace with zero
    scw_parcels.fillna({'ExemptPct':0, 'CreditPct':0, 'GIBTRPct':0, 'LISO':0}, inplace=True)
    #modify the impervious cover with the modifying percentages
    scw_parcels['SCW_IMPERM'] = scw_parcels['SCW_IMPERM'] * (1 - scw_parcels['ExemptPct'])
    scw_parcels['SCW_IMPERM'] = scw_parcels['SCW_IMPERM'] * (1 - scw_parcels['CreditPct'])
    scw_parcels['SCW_IMPERM'] = scw_parcels['SCW_IMPERM'] * (1 - scw_parcels['GIBTRPct'])
    scw_parcels['SCW_IMPERM'] = scw_parcels['SCW_IMPERM'] * (1 - scw_parcels['LISO'])

    #calculate the SCW tax with the final impervious surface area
    scw_parcels['SCW_taxT'] = scw_parcels['SCW_IMPERM'] * 0.025
    # compare the reverse engineered and true total SCW tax values
    print(f"Difference between reverse engineered and true tax values: ${round(scw_parcels['SCW_taxT'].sum() - scw_parcels['SCW_Tax'].sum(),0):,}")

    # calculate the impervious square footage from the NOAA dataset
    scw_parcels['NOAA_SQFT'] = scw_parcels['_sum'] * (3.280712335857215667 * 3.280703045338435153)
    #replace original impervious surface area with the appealed impervious surface area if there is one
    scw_parcels.loc[scw_parcels['AppealImpS'].notna(), 'NOAA_SQFT'] = scw_parcels['AppealImpS']
    #divide the surface area by the overlappping parcels which must be done for condos
    scw_parcels['NOAA_SQFT'] = scw_parcels['NOAA_SQFT'] / scw_parcels['SCW_NUM_OC']
    #for any modifying percentages with NULL, replace with zero
    scw_parcels.fillna({'ExemptPct':0, 'CreditPct':0, 'GIBTRPct':0, 'LISO':0}, inplace=True)
    #modify the impervious cover with the modifying percentages
    scw_parcels['NOAA_SQFT'] = scw_parcels['NOAA_SQFT'] * (1 - scw_parcels['ExemptPct'])
    scw_parcels['NOAA_SQFT'] = scw_parcels['NOAA_SQFT'] * (1 - scw_parcels['CreditPct'])
    scw_parcels['NOAA_SQFT'] = scw_parcels['NOAA_SQFT'] * (1 - scw_parcels['GIBTRPct'])
    scw_parcels['NOAA_SQFT'] = scw_parcels['NOAA_SQFT'] * (1 - scw_parcels['LISO'])

    #calculate the SCW tax with the final impervious surface area
    scw_parcels['SCWtaxNOAA'] = scw_parcels['NOAA_SQFT'] * 0.025
    #print the new tax amount and compare to the original
    print(f"Total tax using NOAA: ${scw_parcels['SCWtaxNOAA'].sum():,}")
    print(f"Difference between new and original: ${scw_parcels['SCWtaxNOAA'].sum() - scw_parcels['SCW_Tax'].sum():,}")

    # calculate the impervious square footage from the NOAA and buildings dataset
    scw_parcels['NOAAwB_SQFT'] = scw_parcels['NwB_sum'] * (3.280712335857215667 * 3.280703045338435153)
    #replace original impervious surface area with the appealed impervious surface area if there is one
    scw_parcels.loc[scw_parcels['AppealImpS'].notna(), 'NOAAwB_SQFT'] = scw_parcels['AppealImpS']
    #divide the surface area by the overlappping parcels which must be done for condos
    scw_parcels['NOAAwB_SQFT'] = scw_parcels['NOAAwB_SQFT'] / scw_parcels['SCW_NUM_OC']
    #for any modifying percentages with NULL, replace with zero
    scw_parcels.fillna({'ExemptPct':0, 'CreditPct':0, 'GIBTRPct':0, 'LISO':0}, inplace=True)
    #modify the impervious cover with the modifying percentages
    scw_parcels['NOAAwB_SQFT'] = scw_parcels['NOAAwB_SQFT'] * (1 - scw_parcels['ExemptPct'])
    scw_parcels['NOAAwB_SQFT'] = scw_parcels['NOAAwB_SQFT'] * (1 - scw_parcels['CreditPct'])
    scw_parcels['NOAAwB_SQFT'] = scw_parcels['NOAAwB_SQFT'] * (1 - scw_parcels['GIBTRPct'])
    scw_parcels['NOAAwB_SQFT'] = scw_parcels['NOAAwB_SQFT'] * (1 - scw_parcels['LISO'])

    #calculate the SCW tax with the final impervious surface area
    scw_parcels['SCWtaxNOAAwB'] = scw_parcels['NOAAwB_SQFT'] * 0.025
    #print the new tax amount and compare to the original
    print(f"Total tax using NOAA and buildings: ${scw_parcels['SCWtaxNOAAwB'].sum():,}")
    print(f"Difference between new plus buildings and original: ${scw_parcels['SCWtaxNOAAwB'].sum() - scw_parcels['SCW_Tax'].sum():,}")

    return

def impervious_comparison():
    '''
    This function compares how well the original LA County impervious dataset captures building footprints compared to the NOAA dataset.
    '''

    #read in building shapefile with pixel counts
    buildings = gpd.read_file('./LACounty_buildings/LACounty_buildings_LAimpervious_NOAAimpervious.shp', engine='pyogrio', columns=['LC_sum','NOAA_sum','area_ft'])
    buildings.dropna(inplace=True, subset=['LC_sum', 'NOAA_sum'])

    # calculate area of in ft estimated by original LA County impervious surface dataset
    buildings['LC_area'] = buildings['LC_sum'] * (0.75 * 0.75)

    # calculate area in ft of estimated by NOAA impervious surface dataset
    buildings['NOAA_area'] = buildings['NOAA_sum'] * (3.280712335857215667 * 3.280703045338435153)

    # calculate LA county dataset ratio
    buildings['la_ratio'] = buildings['LC_area'] / buildings['area_ft']

    # calculate NOAA dataset ratio
    buildings['noaa_ratio'] = buildings['NOAA_area'] / buildings['area_ft']

    print(f"LA County dataset correctly captured building area ratio: {buildings['la_ratio'].mean()}")
    print(f"NOAA dataset correctly captured building area ratio: {buildings['noaa_ratio'].mean()}")

    return

#################
#MAIN
##############

# download the SCW parcels dataset
# download_shp()

# combine the building outlines and NOAA land cover data
# combine_landcover()

# compare the original tax collected to the newly calculated value
comparison()

# compare impervious cover datasets
# impervious_comparison()

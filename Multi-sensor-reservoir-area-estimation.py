#!/usr/bin/env python
# coding: utf-8

import ee 
# import folium
# import geehydro
from datetime import datetime as dt
# from IPython.display import Image

from scipy import stats
import pandas as pd
import time
# import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import plotly.graph_objs as go
import plotly
import fiona
fiona.drvsupport.supported_drivers['KML'] = 'rw'

# initialize the connection to the server
from utility.config import service_account,key_file

credentials = ee.ServiceAccountCredentials(service_account, key_file)
ee.Initialize(credentials)

########################### General helpful functions #####################

# Mapdisplay function is taken directly from free course EEwPython for visualisation purpose 
# def Mapdisplay(center, dicc, Tiles="OpensTreetMap",zoom_start=10):
#     '''
#     :param center: Center of the map (Latitude and Longitude).
#     :param dicc: Earth Engine Geometries or Tiles dictionary
#     :param Tiles: Mapbox Bright,Mapbox Control Room,Stamen Terrain,Stamen Toner,stamenwatercolor,cartodbpositron.
#     :zoom_start: Initial zoom level for the map.
#     :return: A folium.Map object.
#     '''
#     mapViz = folium.Map(location=center,tiles=Tiles, zoom_start=zoom_start)
#     for k,v in dicc.items():
#         if ee.image.Image in [type(x) for x in v.values()]:
#             folium.TileLayer(
#                 tiles = v["tile_fetcher"].url_format,
#                 attr  = 'Google Earth Engine',
#                 overlay =True,
#                 name  = k
#               ).add_to(mapViz)
#         else:
#             folium.GeoJson(
#             data = v,
#             name = k
#           ).add_to(mapViz)
#     mapViz.add_child(folium.LayerControl())
#     return mapViz

# Coverts a polygon geometry object to earth engine feature
def poly2feature(polygon,shp_file_flag):
    if(polygon.type=='MultiPolygon'):
        all_cords=[]
        for poly in polygon.geoms:
            x,y = poly.exterior.coords.xy
            all_cords.append(np.dstack((x,y)).tolist())
        if(shp_file_flag):
            g=ee.Geometry.MultiPolygon(all_cords).buffer(1000)  #buffer for shape file
        else:
            g=ee.Geometry.MultiPolygon(all_cords)#.buffer(2500) in meters  # no buffer for kml file because the polygons are already made with buffer
        
    else:  
        x,y = polygon.exterior.coords.xy
        cords = np.dstack((x,y)).tolist()
        
        if(shp_file_flag):
            g=ee.Geometry.Polygon(cords).buffer(1000)  #buffer for shape file
        else:
            g=ee.Geometry.Polygon(cords)#.buffer(2500) in meters  # no buffer for kml file because the polygons are already made with buffer
        
    feature = ee.Feature(g)
    return feature

# Function to convert feature collection to dictionary.
def fc_2_dict(fc):
    prop_names = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
    reducer=ee.Reducer.toList().repeat(prop_names.size()),
      selectors=prop_names).get('list')

    return ee.Dictionary.fromLists(prop_names, prop_lists)

# Prepares a mosaic using image collection of one day
def day_mosaic(date,imcol,satellite):
    d = ee.Date(date)
    img=imcol.first()
    
    im = imcol.filterDate(d, d.advance(1, "day")).mosaic()
    
    if(satellite=='landsat-08'):
        sun_azimuth=img.get('SUN_AZIMUTH')
        sun_altitude=img.get('SUN_ELEVATION')
    elif(satellite=='sentinel-2'):
        sun_azimuth=img.get('MEAN_SOLAR_AZIMUTH_ANGLE')
        sun_altitude=ee.Number(90).subtract(img.get('MEAN_SOLAR_ZENITH_ANGLE'))
    else:
        return im.set("system:time_start", d.millis(),"DATE_ACQUIRED", d.format("YYYY-MM-dd"))
    
    return im.set("system:time_start", d.millis(),"DATE_ACQUIRED", d.format("YYYY-MM-dd"),
                  "SUN_AZIMUTH",sun_azimuth,"SUN_ALTITUDE",sun_altitude)

# Prepares an image collection of mosaics by date
def mosaicByDate(imcol,satellite):
    # imcol: An image collection
    # returns: An image collection
    
    # Converting collection to list
    imlist = imcol.toList(imcol.size())
    
    # Extracting unique dates in the collection  
    unique_dates = imlist.map(lambda image: ee.Image(image).date().format("YYYY-MM-dd")).distinct()
    
    mosaic_imlist = unique_dates.map(lambda date: day_mosaic(date,imcol,satellite))

    return ee.ImageCollection(mosaic_imlist)

##################### Methods to compute water area for landsat/Sentinel2 #########################

################## Helpful functions ofr methods computing water area ################
# Function to rename bands 
def renameBands_Landsat8(x,product):
    if(product=='SR'):
        bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']
    elif(product=='TOA'):
        bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'BQA']
    elif(product=='S2_SR'):
        bands = ['L_B', 'L_G', 'L_R', 'L_NIR', 'L_SWIR1', 'L_SWIR2', 'qa_pixel']
    new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'qa_pixel']
    return x.select(bands).rename(new_bands)

# Function to create cloud mask
def createCloudAndShadowBand(x,product):
    qa = x.select('qa_pixel');
    if(product=='SR'):
        cloudbitnumber=2**3+2**8+2**9
        cloudshadowbitnumber=2**4+2**10+2**11
    elif(product=='TOA'):
        cloudbitnumber=2**4+2**5+2**6
        cloudshadowbitnumber=2**7+2**8
    elif(product=='S2_SR'):
        cloudbitnumber=2**10
        cloudshadowbitnumber=2**11   #cirrus bit number, to keep similarity with TOA and SR products,it is named cloud shadow bit
    cloudBitMask = ee.Number(cloudbitnumber).int();
    cloudShadowBitMask = ee.Number(cloudshadowbitnumber).int();
    cloud = qa.bitwiseAnd(cloudBitMask).eq(cloudbitnumber);
    cloudShadow = qa.bitwiseAnd(cloudShadowBitMask).eq(cloudshadowbitnumber);
    mask = (ee.Image(0).where(cloud.eq(1), 1)
        .where(cloudShadow.eq(1), 1)
        .rename('cloud_mask'));
    return x.addBands(mask)

########################### Helpful functions for DSWE ################
# INDICES
def calc_mndwi(image):
    mndwi = ee.Image(0).expression(
        '((g - swir1)/(g + swir1)) * 10000',
            {
                'g': image.select("G"),
                'swir1': image.select("SWIR1")
            })
    return mndwi.toInt16().rename("MNDWI")

def calc_mbsr(image):
    mbsr = ee.Image(0).expression(
        '(g + r) - (nir + swir1)',
        {
            'g': image.select("G"),
            'r': image.select("R"),
            'nir': image.select("NIR"),
            'swir1': image.select("SWIR1")
        })
    return mbsr.toInt16().rename("MBSR")

def calc_ndvi(image):
    ndvi = ee.Image(0).expression(
        '((nir - r)/(nir + r)) * 10000',
        {
          'nir': image.select("NIR"),
          'r': image.select("R")
        })
    return ndvi.toInt16().rename("NDVI")

def calc_awesh(image):
    awesh = ee.Image(0).expression(
        'blue + A*g - B*(nir+swir1) - C*swir2',
        {
            'blue': image.select('B'),
            'g': image.select('G'),
            'nir': image.select('NIR'),
            'swir1': image.select('SWIR1'),
            'swir2': image.select('SWIR2'),
            'A': 2.5,
            'B': 1.5,
            'C': 0.25
        })
    return awesh.toInt16().rename("AWESH")

# wrapper
def calc_indices(image):
    bands = ee.Image([
        calc_mndwi(image),
        calc_mbsr(image),
        calc_ndvi(image),
        calc_awesh(image),
        image.select("B"),
        image.select("NIR"),
        image.select("SWIR1"),
        image.select("SWIR2"),
        image.select("cloud_mask")
    ])
    return bands.set('system:time_start', image.get('system:time_start'))

# DSWE test functions
def test1(image):
    return image.select("MNDWI").gt(124)    ## wigt - default value 0.0124 * 10000

def test2(image):
    return image.select("MBSR").gt(0)      ## mbsrv>mbsrn -> mbsr=mbsrv-mbsrn>0

def test3(image):
    return image.select("AWESH").gt(0)    ## awgt - default value 0

def test4(image):
    x = (image.select("MNDWI").gt(-5000)       ## pswt_1_mndwi - default value -0.044 * 10000
        .add(image.select("SWIR1").lt(900))    ## pswt_1_swir1 - default value 900
        .add(image.select("NIR").lt(1500))     ## pswt_1_nir - default value 1500
        .add(image.select("NDVI").lt(7000))     ## pswt_1_ndvi - default value 0.7 * 10000
        )
    return x.eq(4)

def test5(image):
    x = (image.select("MNDWI").gt(-5000)         ## pswt_2_mndwi - default value -0.5 * 10000
        .add(image.select("B").lt(1000))        ## pswt_2_blue  - default value 1000
        .add(image.select("NIR").lt(2500))      ## pswt_2_nir   - default value 2500
        .add(image.select("SWIR1").lt(3000))    ## pswt_2_swir1 - default value 
        .add(image.select("SWIR2").lt(1000))
        )
    return x.eq(5)

def cloudTest(image):
    return image.select('cloud_mask').eq(1)

# wrapper/multiplier function
def addTests(image):
    x1 = test1(image)
    x2 = test2(image).multiply(10);
    x3 = test3(image).multiply(100);
    x4 = test4(image).multiply(1000);
    x5 = test5(image).multiply(10000);
    cld = cloudTest(image);
    res = x1.add(x2).add(x3).add(x4).add(x5).rename('test')         .where(cld.eq(1), -1)         .set('system:time_start', image.get('system:time_start'));
    return res

# DSWE CLASSES
def isDSWE0(image):
    y1 = image.lte(10).add(image.gte(0)).eq(2)
    y2 = image.eq(100).add(image.eq(1000)).eq(1)
    y = y1.add(y2).gt(0)         .rename("DSWE0")         .set('system:time_start', image.get('system:time_start'))
    return y

def isDSWE1(image):
    y1 = image.gte(11101).add(image.lte(11111)).eq(2)
    y2 = image.eq(1111).add(image.eq(10111)).add(image.eq(11011)).eq(1)
    y = y1.add(y2).gt(0)         .rename("DSWE1")         .set('system:time_start', image.get('system:time_start'))
    return y

def isDSWE2(image):
    y1 = image.eq(111).add(image.eq(1011)).add(image.eq(1101)).add(image.eq(1110)).add(image.eq(10011))          .add(image.eq(10101)).add(image.eq(10110)).add(image.eq(11001)).add(image.eq(11010)).add(image.eq(11100)).eq(1)
    y = y1.gt(0)         .rename("DSWE2")         .set('system:time_start', image.get('system:time_start'))
    return y

def isDSWE3(image):
    y = image.eq(11000)         .rename("DSWE3")         .set('system:time_start', image.get('system:time_start'))
    return y
    
def isDSWE4(image):
    y1 = image.eq(11).add(image.eq(101)).add(image.eq(110)).add(image.eq(1001)).add(image.eq(1010))          .add(image.eq(1100)).add(image.eq(10000)).add(image.eq(10001)).add(image.eq(10010)).add(image.eq(10100)).eq(1)
    y = y1.gt(0)         .rename("DSWE4")         .set('system:time_start', image.get('system:time_start'))
    return y

def isDSWE9(image):
    y = image.eq(-1)         .rename("DSWE9")         .set('system:time_start', image.get('system:time_start'))
    return y

#################### DSWE Function #########################
def dswe(image,product):
    '''
    DSWE
    ====
    Apply DSWE algorithm to a single image
    Arguments:
    ----------
    image:  ee.Image object (must be Landsat-8 SR Collection-2 TR-1 product)
    product: 'SR','TOA' or 'S2_SR'
    '''
    # Reading a DEM
    dem=ee.Image('CGIAR/SRTM90_V4').select('elevation')
    
    # Add Hill shade
    sun_altitude=image.get('SUN_ALTITUDE')
    sun_azimuth=image.get('SUN_AZIMUTH')
    image=image.addBands(ee.Terrain.hillshade(dem,sun_azimuth,sun_altitude).rename('hillshade'))
    
    # Calculating slope
    slope=ee.Terrain.slope(dem).rename('slope')
    
    # Add cloud mask dnd rename bands
    img = createCloudAndShadowBand(renameBands_Landsat8(image,product),product)
    
    # Calculate indices
    indices = calc_indices(img)
    
    # Perform comparisons of various indices with thresholds and outputs the result of each test in a bit
    tests = addTests(indices)
    
    # Classify pixels into different classes to create interpreted dswe band
    dswe = ee.Image(-1)         .where(isDSWE0(tests), 0)         .where(isDSWE1(tests), 1)         .where(isDSWE2(tests), 2)         .where(isDSWE3(tests), 3)         .where(isDSWE4(tests), 4)         .where(isDSWE9(tests), 9)         .updateMask(img.select('qa_pixel').mask())         .rename("DSWE") 
    
    # Classifying pixels having  hill shade less than equal to 110 as not water(0) 
    dswe=dswe.where(image.select('hillshade').lte(110),0)
    
    # Classifying pixels using interpreted DSWE and slope
    dswe=dswe.where((dswe.eq(4) and slope.gte(5.71)).Or                # 10% slope = 5.71°
                      (dswe.eq(3) and slope.gte(11.31)).Or           # 20% slope = 11.31°
                      (dswe.eq(2) and slope.gte(16.7)).Or            # 30% slope = 16.7°
                      (dswe.eq(1) and slope.gte(16.7)), 0);          # 30% slope = 16.7°

    return dswe

########################### MNDWI Function ####################
def mndwi(image,product):
    if(product=='SR'):
        img = createCloudAndShadowBand(renameBands_Landsat8(image,product),product)
    elif(product=='S2_SR'):
        img = createCloudAndShadowBand(image,product)
    mndwi = img.normalizedDifference(['G', 'SWIR1']).rename("MNDWI")
    return mndwi.addBands(img.select('cloud_mask'))

########################## NDWI Function ####################
def ndwi(image,product):
    if(product=='SR'):
        img = createCloudAndShadowBand(renameBands_Landsat8(image,product),product)
    elif(product=='S2_SR'):
        img = createCloudAndShadowBand(image,product)
    ndwi = img.normalizedDifference(['G', 'NIR']).rename("NDWI")
    return ndwi.addBands(img.select('cloud_mask'))

################# Landsat-8 function ###################

def landsat_water_area_calculation_in_image(inp_image,method,clip_feature=None,product='SR',image_return=False):
    # Clipping image
    if(clip_feature):
        pro_image=inp_image.clip(clip_feature)
    else:
        pro_image=inp_image
        
    # Calculating method
    if(method=='NDWI'):
        pro_image=ndwi(pro_image,product)
        # Removing land pixels based on ndwi<=-0.05
        water_classified=pro_image.select('NDWI').gt(0)
        cloud_classified=pro_image.select('cloud_mask').eq(1)
    elif(method=='MNDWI'):
        pro_image=mndwi(pro_image,product)
        # Removing land pixels based on mndwi<=0
        water_classified=pro_image.select('MNDWI').gt(0)
        cloud_classified=pro_image.select('cloud_mask').eq(1)
    elif(method=='DSWE'):
        pro_image=dswe(pro_image,product)
        # Removing land pixels based on dswe=0 & 9
        water_classified=pro_image.gt(0).add(pro_image.lte(4)).eq(2)
        cloud_classified=pro_image.eq(9)
    
    # Returning water image if yes
    if(image_return):
        if(method=='DSWE'):
            return pro_image.where(water_classified.eq(1),1).where(water_classified.eq(0),0).updateMask(pro_image.neq(9))
        else:
            return pro_image.select(method).where(water_classified.eq(1),1).where(water_classified.eq(0),0).updateMask(
                                                                                    pro_image.select('cloud_mask').neq(1))
    # Total pixels    
    total_pixels=ee.Number(pro_image.select(method).reduceRegion(reducer=ee.Reducer.count(),geometry=clip_feature.geometry(),
                                                                      scale=30).get(method))
    
    # Counting cloud pixels 
    cloud_image=pro_image.select(method).updateMask(cloud_classified)
    cloud_pixels=ee.Number(cloud_image.reduceRegion(reducer=ee.Reducer.count(),geometry=clip_feature.geometry(),
                                                                      scale=30).get(method))
    
    # Counting water pixels and calculating area
    water_image=pro_image.select(method).updateMask(water_classified)
    water_pixels=ee.Number(water_image.reduceRegion(reducer=ee.Reducer.count(),geometry=clip_feature.geometry(),
                                                                      scale=30).get(method))
    
    return ee.Feature(None,{'Satellite':'Landsat-08_'+method,
                            'Date':inp_image.get('DATE_ACQUIRED'),
                            'Water Area':water_pixels.multiply(30).multiply(30).divide(1000000),
                            'Total Area':total_pixels.multiply(30).multiply(30).divide(1000000),
                            'Cloud Percent Area':cloud_pixels.divide(total_pixels).multiply(100)})

############### Sentinel-1 Functions #################

# Filters speckle noise
def Specklefilter(image):
    vv = image.select('VV') #select the VV polarization band
    vv_smoothed = vv.focal_median(30,'square','meters').rename('VV_Filtered') #Apply a focal median filter
    return image.addBands(vv_smoothed) #Add filtered VV band to original image

# Reservoir Water area calculation for sentinel-1 data
def sentinel1_water_area_calculation_in_image(inp_image,clip_feature=None,image_return=False):
    # Speckle filter 
    pro_image=Specklefilter(inp_image)
    # clipping image if required
    if(clip_feature):
        pro_image=pro_image.clip(clip_feature)
    pro_image=pro_image.select('VV_Filtered')
    # Removing land pixels based on VV>=-13 dB
    water_classified=pro_image.lt(-13)
    
    # Returning image if yes
    if(image_return):
        return pro_image.where(water_classified.eq(1),1).where(water_classified.eq(0),0)
    
    # Counting total pixels, water pixels and calculating area
    # Total pixels    
    total_pixels=ee.Number(pro_image.reduceRegion(reducer=ee.Reducer.count(),
                                                          geometry=clip_feature.geometry(),scale=30).get('VV_Filtered'))
    # Water pixels
    pro_image=pro_image.updateMask(water_classified)
    water_pixels=ee.Number(pro_image.reduceRegion(reducer=ee.Reducer.count(),
                                                          geometry=clip_feature.geometry(),scale=30).get('VV_Filtered'))
    return ee.Feature(None,{'Satellite':'Sentinel-1',
                            'Product': 'S1_GRD',
                            'Date':inp_image.get('DATE_ACQUIRED'),
                            'Water Area':water_pixels.multiply(30).multiply(30).divide(1000000),
                            'Total Area':total_pixels.multiply(30).multiply(30).divide(1000000),
                             'Cloud Percent Area': ee.Number(0)})

################## Sentinel-2 Functions ###############

# Having a uniform resolution in sentinel image
def sentinel2_refineresolution(image,band,scale):
    band_img=image.select(band)
    image=image.resample('bilinear').reproject(**{
      'crs': band_img.projection().crs(),
      'scale': scale
        });
    return image

# Renaming bands 
def renameBands_sentinel2(x,product):
    if(product=='S2_SR'):
        bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'QA60']
    new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'qa_pixel']
    return x.select(bands).rename(new_bands)

# Transforming Sentinel-2 to Landsat-8
def sentinel2_to_landsat8(image,product):
    image=renameBands_sentinel2(image,product)
    # Linear transformations
    landsat_b=image.select('B').multiply(0.9570).add(0.0003).rename('L_B')
    landsat_g=image.select('G').multiply(1.0304).add(0.0015).rename('L_G')
    landsat_r=image.select('R').multiply(0.9533).add(0.0041).rename('L_R')
    landsat_nir=image.select('NIR').multiply(1.0157).add(0.0139).rename('L_NIR')
    landsat_swir1=image.select('SWIR1').multiply(0.9522).add(0.0034).rename('L_SWIR1')
    landsat_swir2=image.select('SWIR2').multiply(0.9711).add(0.0004).rename('L_SWIR2')
    return image.addBands([landsat_b,landsat_g,landsat_r,landsat_nir,landsat_swir1,landsat_swir2])
    
# Water area calculation in sentinel-2 image
def sentinel2_water_area_calculation_in_image(inp_image,method,clip_feature=None,product='S2_SR',image_return=False):
    # Clipping image
    if(clip_feature):
        pro_image=inp_image.clip(clip_feature)
    else:
        pro_image=inp_image
    
    # Sentinel2 to Landsat8
    pro_image = sentinel2_to_landsat8(pro_image,product)
    
    # Calculating method
    if(method=='NDWI'):
        pro_image=ndwi(pro_image,product)
        # Removing land pixels based on ndwi<=0
        water_classified=pro_image.select('NDWI').gt(0)
        cloud_classified=pro_image.select('cloud_mask').eq(1)
    elif(method=='MNDWI'):
        pro_image=mndwi(pro_image,product)
        # Removing land pixels based on mndwi<=0
        water_classified=pro_image.select('MNDWI').gt(0)
        cloud_classified=pro_image.select('cloud_mask').eq(1)
    elif(method=='DSWE'):
        pro_image=dswe(pro_image,product)
        # Removing land pixels based on dswe=0 & 9
        water_classified=pro_image.gt(0).add(pro_image.lte(4)).eq(2)
        cloud_classified=pro_image.eq(9)
    
    # Returning water image if yes
    if(image_return):
        if(method=='DSWE'):
            return pro_image.where(water_classified.eq(1),1).where(water_classified.eq(0),0).updateMask(pro_image.neq(9))
        else:
            return pro_image.select(method).where(water_classified.eq(1),1).where(water_classified.eq(0),0).updateMask(
                                                                                    pro_image.select('cloud_mask').neq(1))
    # Total pixels    
    total_pixels=ee.Number(pro_image.select(method).reduceRegion(reducer=ee.Reducer.count(),geometry=clip_feature.geometry(),
                                                                      scale=60).get(method))
    # Counting cloud pixels 
    cloud_image=pro_image.select(method).updateMask(cloud_classified)
    cloud_pixels=ee.Number(cloud_image.reduceRegion(reducer=ee.Reducer.count(),geometry=clip_feature.geometry(),
                                                                      scale=60).get(method))
    
    # Counting water pixels and calculating area
    water_image=pro_image.select(method).updateMask(water_classified)
    water_pixels=ee.Number(water_image.reduceRegion(reducer=ee.Reducer.count(),geometry=clip_feature.geometry(),
                                                                      scale=60).get(method))
    
    return ee.Feature(None,{'Satellite':'Sentinel-2_'+method,
                            'Date':inp_image.get('DATE_ACQUIRED'),
                            'Water Area':water_pixels.multiply(60).multiply(60).divide(1000000),
                            'Total Area':total_pixels.multiply(60).multiply(60).divide(1000000),
                            'Cloud Percent Area':cloud_pixels.divide(total_pixels).multiply(100)})

###################################### Main Script ############################

########################## Local computation #################################

def calculate_time_series(reservoir, start_date, end_date, cloud_cover_percent=20):
    
    ############# User Input ##############
    reservoir_name=reservoir
    inp_start_date=start_date
    inp_end_date=end_date
    unique_str=reservoir_name+'_'+inp_start_date+'_'+inp_end_date
    sentinel_1=True
    sentinel_2=True
    landsat_08=True
    products=['SR']               # used for Landsat-08
    methods=['DSWE','NDWI','MNDWI']     # used for Sentinel-2 and Landsat-08
    cloud_percent=cloud_cover_percent          #used for Landsat-08 and Sentinel-2

    ############# Reading Reservoir shapefiles and boundary polygons ###########
    print('Reading reservoir data ......')
    # Filepath to Unmonitored Reservoirs polygon KML file
    res_poly_file = "reservoir_data/Unmonitored_texas_reservoirs.kml"

    # Filepath to Major Reservoirs shapefile
    res_shp_file = "reservoir_data/Major_texas_reservoirs.kml"

    # Reading shapefile and verify if reservoir data is there
    from_shp_file=True  #flag for reservoir from shp file or kml file
    poly_reservoirs=gpd.read_file(res_shp_file, driver='KML')
    name_column='Name'
    if (~(poly_reservoirs['Name'].str.contains(reservoir_name).any())):
        # Reading data from kml file if reservoir data is not in shp file
        poly_reservoirs=gpd.read_file(res_poly_file, driver='KML')
        from_shp_file=False
        name_column='Name'

    # Calculating reservoir centeroid for lat-lon
    poly_reservoirs['Center_point'] = poly_reservoirs['geometry'].to_crs('+proj=cea').centroid.to_crs(
                                                                                poly_reservoirs['geometry'].crs)

    # Extracting reservoir information 
    reservoir_data=poly_reservoirs[poly_reservoirs[name_column]==reservoir_name].reset_index(drop=True)
    res_lat=reservoir_data['Center_point'].y[0]
    res_lon=reservoir_data['Center_point'].x[0]
    res_bbox=reservoir_data['geometry'].bounds

    ######################### Earth Engine Computation ####################
    print('Calculating Water Area time series ......')
    clipping_feature=poly2feature(reservoir_data.geometry[0],from_shp_file)
    # setting the Area of Interest (AOI)
    Reservoir_AOI = ee.Geometry.Rectangle(res_bbox.values.reshape(-1).tolist())
    water_area_list=ee.List([])
    cloud_filter=[ee.Filter.gt('Cloud Percent Area',0),ee.Filter.lte('Cloud Percent Area',cloud_percent)]

    if(landsat_08):
        for product in products:
            if(product=='SR'):
                landsat_collection_id="LANDSAT/LC08/C02/T1_L2"
            elif(product=='TOA'):
                landsat_collection_id="LANDSAT/LC08/C01/T1_TOA"
            # filter area and dates
            landsat_AOI = ee.ImageCollection(landsat_collection_id).filterBounds(Reservoir_AOI).filterDate(inp_start_date,inp_end_date)
            # make a mosaic if needed
            landsat_AOI_mosaic = mosaicByDate(landsat_AOI,'landsat-08')
            for method in methods:
                # water area calculation
                landsat_water_area_collection=landsat_AOI_mosaic.map(lambda image: 
                                                landsat_water_area_calculation_in_image(image,method,clipping_feature,product))
                # Adding water area collection to final list
                water_area_list=water_area_list.add(landsat_water_area_collection)
                #temp_df=pd.DataFrame(fc_2_dict(landsat_water_area_collection.filter(cloud_filter)).getInfo())
                #result_df=result_df.append(temp_df)
        print('Calculated data for Landsat-08.')

    if(sentinel_1):
        # filter area and dates
        sentinel_1_AOI = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(Reservoir_AOI).filterDate(inp_start_date,
                                                                                                        inp_end_date)
        # make a mosaic if needed
        sentinel_1_AOI_mosaic=mosaicByDate(sentinel_1_AOI,'sentinel-1')
        # water area calculation
        sentinel_1_water_area_collection=sentinel_1_AOI_mosaic.map(lambda image: sentinel1_water_area_calculation_in_image(image,
                                                                                                                clipping_feature))
        # Adding water area collection to final list
        water_area_list=water_area_list.add(sentinel_1_water_area_collection)
        #temp_df=pd.DataFrame(fc_2_dict(sentinel_1_water_area_collection.filter(cloud_filter)).getInfo())
        #result_df=result_df.append(temp_df)
        print('Calculated data for Sentinel-1.')

    if(sentinel_2):
        # filter area and dates
        sentinel_2_AOI = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(Reservoir_AOI).filterDate(inp_start_date,
                                                                                                       inp_end_date)
        #Refining resolution
        sentinel_2_AOI = sentinel_2_AOI.map(lambda image : sentinel2_refineresolution(image, 'B1', 60))
        # make a mosaic if needed
        sentinel_2_AOI_mosaic=mosaicByDate(sentinel_2_AOI,'sentinel-2')
        for method in methods:
            # water area calculation
            sentinel2_water_area_collection=sentinel_2_AOI_mosaic.map(lambda image: 
                                        sentinel2_water_area_calculation_in_image(image,method,clipping_feature))
            # Adding water area collection to final list
            water_area_list=water_area_list.add(sentinel2_water_area_collection)
            #temp_df=pd.DataFrame(fc_2_dict(sentinel2_water_area_collection.filter(cloud_filter)).getInfo())
            #result_df=result_df.append(temp_df)
        print('Calculated data for Sentinel-2.')
        
    # Merging/Flattening all water area collections
    resulting_water_area_collection=ee.FeatureCollection.flatten(ee.FeatureCollection(water_area_list))

    # Converting the feature collection to dataframe
    print('Exporting Water Area Data to Pandas Dataframe ....')
    result_df=pd.DataFrame(fc_2_dict(resulting_water_area_collection.filter(cloud_filter)).getInfo())
    
    if(len(result_df)>3):
        result_df['Date']=pd.to_datetime(result_df['Date'])
        result_df['Week']=result_df['Date'].dt.isocalendar().week
        result_df['Year']=result_df['Date'].dt.year
        #result_df=result_df[result_df['Cloud Percent Area']<=cloud_percent][result_df['Water Area']>0]
        result_df['Water Area Z']=stats.zscore(result_df['Water Area'])

        filtered_df=result_df[abs(result_df['Water Area Z'])<=1.2]

        water_ts_df=filtered_df.groupby(['Year','Week']).agg({'Water Area':['mean','min','max','count'],'Date':['first']})
        water_ts_df = water_ts_df.reset_index(level=['Year',"Week"])
        water_ts_df.columns = [f'{i}_{j}' if j != '' else f'{i}' for i,j in water_ts_df.columns]
        water_ts_df=water_ts_df.sort_values(by='Date_first')


        fig = go.Figure([
            go.Scatter(
                name='Water Area',
                x=water_ts_df['Date_first'],
                y=water_ts_df['Water Area_mean'],
                mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
            ),
            go.Scatter(
                name='Upper Bound',
                x=water_ts_df['Date_first'],
                y=water_ts_df['Water Area_max'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Lower Bound',
                x=water_ts_df['Date_first'],
                y=water_ts_df['Water Area_min'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(10, 15, 75, 0.2)',
                fill='tonexty',
                showlegend=False
            )
        ])
        fig.update_layout(
            yaxis_title='Water Area (Km<sup>2</sup>)',
            xaxis_title='Date',
            title='Water Area time series for '+reservoir_name,
            title_font={'color':'red', 'size':20},
            hovermode="x",
            #yaxis_range=(1.5,4.5),
            showlegend=False
        )
        fig.update_xaxes(rangeslider_visible=True)
        #fig.show()
        

        ######################## Exporting Results from Earth Engine to Drive ###############
        # # Export the Water Area FeatureCollection to a CSV file.
        # print('Exporting Water Area CSV file ......')
        # task = ee.batch.Export.table.toDrive(**{
        #   'collection': resulting_water_area_collection,
        #   'description': unique_str,
        #   'fileFormat': 'CSV'
        # })
        # task.start()
        # while task.active():
        #     print('Polling for task (id: {}).'.format(task.id))
        #     time.sleep(5)
        # print('Exported successfully!')
        
        return (plotly.offline.plot(fig,include_plotlyjs=False,output_type='div'),[res_lon,res_lat])
        
    else:
        return ('<div> There is not enough data to create time series. Please select a longer time period for time series plot. <div>')
    
#print(calculate_time_series('Cox Lake / Raw Water Lake / Recycle Lake','2018-01-01','2020-12-31',20))  #test case

import pickle
import shapely
import numpy as np
import geopandas as gpd
import pandas as pd

from geovoronoi import voronoi_regions_from_coords


def pickleLoad(fname):
    if len(fname) > 0:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        return []
    

def getVoronoiPolygons(xy, city_census_tracts, column):

    bbox = get_bbox(xy)
    node_ids = city_census_tracts.index
    region_polys, region_pts = voronoi_regions_from_coords(xy, bbox)
    region_pts = dict((k, node_ids[v[0]]) for k, v in region_pts.items())

    voronoi_polygons = gpd.GeoDataFrame({'geometry': region_polys.values()}, index=region_polys.keys())
    voronoi_polygons['nodes'] = pd.Series(region_pts)
    voronoi_polygons = voronoi_polygons.join(city_census_tracts[column], on='nodes')

    return voronoi_polygons


def get_bbox(xy, scale=1.5):

    min_x, min_y = np.min(xy, axis=0)
    max_x, max_y = np.max(xy, axis=0)
    
    k = (max_y - min_y) * scale - (np.abs(min_x) - np.abs(max_x))
    k = k/2
    
    max_x += k
    min_x -= k
    bbox_polygon = shapely.geometry.Polygon([
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y)
    ])

    return bbox_polygon
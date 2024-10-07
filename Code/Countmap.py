# %%
import geopandas as gpd
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import folium
from folium.plugins import HeatMap
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import morecantile
import mercantile
from pyproj import Transformer
# from pyproj import Proj, transform
import time
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from osgeo import osr
from shapely import geometry, wkt
from shapely.ops import transform

def set_up_connection(db_host, port, user, pwd, db):
    try:
        connection = psycopg2.connect(
        host=db_host,
        port=port,
        user=user,
        password=pwd,
        database=db,
        )
        # cursor = connection.cursor()
        print('Connection established successfully!')
        return connection
    except Exception as e:
        print("Error connecting to the database:", e)

def sql_to_geodataframe(query, conn):
   # crs: Coordinate reference system to use for the returned GeoDataFrame
   pd.set_option('display.max_colwidth', None)
   geo_df = gpd.read_postgis(query, conn, geom_col='geom_centroid', crs="3413")
   return geo_df

def lat_long_to_tile(lat, long, zoom_level):
    # Set default TileMatrixSets into Spherical Mercator  (EPGS:4326)
    tms = morecantile.tms.get("WGS1984Quad")
    # Get Tile X, Y, Z from lat and long
    Tile_XYZ = tms.tile(lng = long, lat = lat, zoom = zoom_level)
    print("Tile information:" + str(Tile_XYZ))
    return Tile_XYZ

def tile_to_BBOX(Tile_XYZ):
    tms = morecantile.tms.get("WGS1984Quad")
    bbox = tms.bounds(Tile_XYZ)
    # bbox[0]: long_xmin, bbox[1]: lat_ymin, bbox[2]: long_xmax, bbox[3]: lat_ymax
    return BBOX_transformation(bbox[0], bbox[1], bbox[2], bbox[3])

def lat_long_to_BBOX(lat, long, zoom_level):
    Tile_XYZ = lat_long_to_tile(lat, long, zoom_level)
    return tile_to_BBOX(Tile_XYZ)

def BBOX_transformation(x1, y1, x2, y2):
    # x1: xmin, y1: xmin, x2: xmax, y2: ymax
    p1 = geometry.Point(x1, y1)
    p2 = geometry.Point(x1, y2)
    p3 = geometry.Point(x2, y2)
    p4 = geometry.Point(x2, y1)
    pointList = [p1, p2, p3, p4]
    poly = geometry.Polygon([[p.x, p.y] for p in pointList])

    # If we want to transform BBOX into 3413 first, then perform the following code
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    reprojected_BBOX = transform(transformer.transform, poly)
    return str(reprojected_BBOX)

def get_centroids_within_bbox(bbox, col_names, conn):
    # If we want to query data from database in 3413, perform this
    query = "SELECT " + col_names + " FROM alaska_all_3413 polygons WHERE ST_within(polygons.geom_centroid, ST_Buffer(ST_GeomFromText('" + bbox + "', 3413), 100))"
    
    # If we want to query data from database in 4326, perform this
    # query = "SELECT " + col_names + " FROM alaska_146_157_167_168_v2 polygons WHERE ST_within(ST_centroid(polygons.geom), ST_GeomFromText('" + bbox + "',4326))"
    # print("--------------------The query is:--------------------")
    print(query)
    start = time.time()
    geo_df = sql_to_geodataframe(query, conn)
    end = time.time()
    print(f"Data Extracted! The runtime of getting data from database is {end - start}")
    return geo_df

def lat_long_to_geo_df(lat, long, zoom_level, col_names, conn):
    reprojected_bbox = lat_long_to_BBOX(lat, long, zoom_level)
    return get_centroids_within_bbox(reprojected_bbox, col_names, conn)

def geo_df_to_heatmap_xy(geo_df):
    points = geo_df.copy()
    # change geometry 
    points['geom_centroid'] = points['geom_centroid']
    # If database is in 3413, perfom the following code
    points['geom_centroid'] = points['geom_centroid'].to_crs(epsg=4326)
    # print(points)
    longs = [point.x for point in points.geom_centroid]
    lats = [point.y for point in points.geom_centroid]
    return longs, lats

def get_data_extent(longs, lats, tile_extent):
    if not longs or not lats:
        # Each tile has 256 pixels, and each pixel is 0.00390625 in bin width and height
        bin_x = np.arange(tile_extent[0], tile_extent[1], 0.00390625)
        bin_y = np.arange(tile_extent[2], tile_extent[3], 0.00390625)
    else:
        min_long = min(longs)
        max_long = max(longs)
        min_lat = min(lats)
        max_lat = max(lats)
        data_extent = [min_long, max_long, min_lat, max_lat]
        # To make sure 0 value can fill into the no data region
        bin_x = np.arange(min(tile_extent[0], data_extent[0]), max(tile_extent[1], data_extent[1]), 0.00390625)
        bin_y = np.arange(min(tile_extent[2], data_extent[2]), max(tile_extent[3], data_extent[3]), 0.00390625)
    return [bin_x, bin_y]

def extent_adjustment(tile):
    tms = morecantile.tms.get("WGS1984Quad")
    tile_extent = [tms.xy_bounds(tile)[0], tms.xy_bounds(tile)[2], tms.xy_bounds(tile)[1], tms.xy_bounds(tile)[3]]
    print("Tile extent:")
    print(tile_extent[0], tile_extent[1], tile_extent[2], tile_extent[3])
    # bin_x = np.arange(tile_extent[0], tile_extent[1], 0.0001)
    # bin_y = np.arange(tile_extent[2], tile_extent[3], 0.0001)
    # return [bin_x, bin_y]
    return tile_extent

def create_heatmap(longs, lats, tile, data_bins, extent):
    heatmap, xedges, yedges = np.histogram2d(longs, lats, bins=data_bins, density=False)
    # heatmap = gaussian_filter(heatmap, sigma=32, radius=20)
    data_extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    # Clip the heatmap based on the tile extent
    heatmap_clipped, xedges_clipped, yedges_clipped = clip_heatmap(heatmap, extent, xedges, yedges)
    data_extent_clipped = [xedges_clipped[0], xedges_clipped[-1], yedges_clipped[0], yedges_clipped[-1]]

    print("Heatmap extent:")
    print(data_extent[0], data_extent[1], data_extent[2], data_extent[3])
    print("Clipped heatmap extent:")
    print(data_extent_clipped[0], data_extent_clipped[1], data_extent_clipped[2], data_extent_clipped[3])
    return heatmap_clipped.T, tile

def clip_heatmap(heatmap, extent, xedges, yedges):
    xmin, xmax, ymin, ymax = extent
    x_indices = np.where((xedges >= xmin) & (xedges <= xmax))[0]
    y_indices = np.where((yedges >= ymin) & (yedges <= ymax))[0]

    # Ensure that the indices don't exceed the size of the histogram
    x_indices = np.clip(x_indices, 0, heatmap.shape[1])
    y_indices = np.clip(y_indices, 0, heatmap.shape[0])

    xedges_clipped = xedges[x_indices[0]:x_indices[-1] + 1]
    yedges_clipped = yedges[y_indices[0]:y_indices[-1] + 1]

    clipped_heatmap = heatmap[y_indices[0]:y_indices[-1], x_indices[0]:x_indices[-1]]
    return clipped_heatmap, xedges_clipped, yedges_clipped

def statistics_histogram(array):
    # Calculate the overall statistics for all heatmaps
    overall_min = np.min(array)
    overall_max = np.max(array)
    overall_mean = np.mean(array)
    overall_std = np.std(array)

    # Print the overall statistics
    print("Overall Min:", overall_min)
    print("Overall Max:", overall_max)
    print("Overall Mean:", overall_mean)
    print("Overall Standard Deviation:", overall_std)

    # Plot histogram
    plt.hist(array, bins=100)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Combined Array')
    plt.savefig("/home/xchen/code/shpMapping/histogram/Tiles_LargerROI.png")
    # plt.savefig("/home/xchen/code/shpMapping/histogram/9Tiles_" + str(tile) + ".png")

    return overall_max

def plot_heatmap(img, tile_extent, tile, max_value):
    print("Plotting heatmap...")
    plt.figure(figsize=(10, 10))
    plt.imshow(img, extent=tile_extent, origin='lower', cmap=cm.jet, vmin=0, vmax=max_value)
    plt.title("Heatmap: " + str(tile))
    plt.xlim(tile_extent[0], tile_extent[1])
    plt.ylim(tile_extent[2], tile_extent[3])
    plt.savefig("/home/xchen/code/shpMapping/heatmap_output/" + str(tile) + ".png")
    plt.close()

def plot_tile(tile, col_names, conn):
    bbox = tile_to_BBOX(tile)
    geo_df = get_centroids_within_bbox(bbox, col_names, conn)
    longs, lats = geo_df_to_heatmap_xy(geo_df)
    tile_extent = extent_adjustment(tile)
    data_bins = get_data_extent(longs, lats, tile_extent)
    data, tile_name = create_heatmap(longs, lats, tile, data_bins, tile_extent)

    # Concatenate all arrays from the list into one single array
    combined_array = np.concatenate(data, axis=None)
    print(combined_array)
    max_value = statistics_histogram(combined_array)
    print(f"The maximum value is: {max_value}")
    print(data, tile_extent)

    plt.figure(figsize=(10, 10))
    plt.imshow(data, extent=tile_extent, origin='lower', cmap=cm.jet, vmin=0, vmax=max_value)
    plt.title("Heatmap: " + str(tile_name))
    plt.xlim(tile_extent[0], tile_extent[1])
    plt.ylim(tile_extent[2], tile_extent[3])
    plt.savefig("/home/xchen/code/shpMapping/heatmap_output/r32r20_buffer_20_"  + str(tile_name) + ".png")
    return 0

def center_tile_to_9_tiles(tile, col_names, center_tile_name, conn):
    tms = morecantile.tms.get("WGS1984Quad")
    data_list = []
    tile_extent_list = []
    tile_name_list = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            t = morecantile.Tile(x=tile.x + j, y=tile.y + i, z=tile.z)
            b = tile_to_BBOX(t)
            geo_df = get_centroids_within_bbox(b, col_names, conn)
            longs, lats = geo_df_to_heatmap_xy(geo_df)
            tile_extent = extent_adjustment(t)
            # To get the data extent after buffering
            data_bins = get_data_extent(longs, lats, tile_extent)
            img, tile_name = create_heatmap(longs, lats, t, data_bins, tile_extent)
            data_list.append(img)
            tile_extent_list.append(tile_extent)
            tile_name_list.append(tile_name)

    # Concatenate all arrays from the list into one single array
    combined_array = np.concatenate(data_list, axis=None)
    print(combined_array)

    max_value = statistics_histogram(combined_array)
    print(f"The maximum value is: {max_value}")

    fig, axs = plt.subplots(3, 3, figsize=(14, 14))

    for ax, data, extent, tilename in zip(axs.flatten(), data_list, tile_extent_list, tile_name_list):
        print(data, extent)
        ax.imshow(data, extent=extent, origin='lower', cmap=cm.jet, vmin=0, vmax=max_value)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(tilename)

    plt.savefig("/home/xchen/code/shpMapping/heatmap_output/" + "Tiles_s32r20_buffer100_" + center_tile_name + "_v5.png")
    return 0

def mouse_to_9_tiles(lat, long, zoom_level, col_names, conn):
    tile_center = lat_long_to_tile(lat, long, zoom_level)
    print(tile_center)
    print(type(tile_center))
    return center_tile_to_9_tiles(tile_center, col_names, str(tile_center), conn)

def plot_all_tiles_within_ROI(ROI_xmin, ROI_xmax, ROI_ymin, ROI_ymax, zoom_level, col_names, conn):
    # Get all tiles within bbox
    tms = morecantile.tms.get("WGS1984Quad")
    # Print tiles that overlap or contain a lng/lat point, bounding box, or GeoJSON objects.
    tiles = tms.tiles(ROI_xmin, ROI_ymin, ROI_xmax, ROI_ymax, zoom_level)
    tiles_list = list(tiles)
    print(tiles_list)
    print(f"The number of tiles within the geographic bbox is: {len(tiles_list)}")
    data_list = []
    tile_extent_list = []
    tile_name_list = []
    
    for tile in tiles_list:
        # Get all points within the tile
        bbox = tile_to_BBOX(tile)
        geo_df = get_centroids_within_bbox(bbox, col_names, conn)
        longs, lats = geo_df_to_heatmap_xy(geo_df)

        # Smoothing each plot
        tile_extent = extent_adjustment(tile)
        data_bins = get_data_extent(longs, lats, tile_extent)
        data, tile_name = create_heatmap(longs, lats, tile, data_bins, tile_extent)
        data_list.append(data)
        tile_extent_list.append(tile_extent)
        tile_name_list.append(tile_name)
    
    max_value = statistics_histogram(np.concatenate(data_list, axis=None))
    
    # Plot all heatmaps after getting the maximum value
    for data, tilename, tile_extent in zip(data_list, tile_name_list, tile_extent_list):
        plot_heatmap(data, tile_extent, tilename, max_value)

    return 0

if __name__=="__main__": 
    hostname = "cici.lab.asu.edu"
    port = "5432"
    user = "postgres"
    password = "shirly"
    database = "postgres"

    column = ["gid", "class", "sensor", "date", "time", "image", "area", "centroidx", "centroidy", "permeter", "length", "width", "geom_centroid"]
    column_names = "centroidx, centroidy, geom_centroid"

    mouse_lat = 71.31610
    mouse_long = -156.60027

    # Small ROI
    # west = -156.47441
    # south = 70.40887
    # east = -153.92505
    # north = 70.87147

    # Large ROI
    # west = -166.43910
    # south = 69.09646
    # east = -141.05628
    # north = 70.95322

    zoom_level = 10
    # heatmap_name = "zoomlevel" + str(zoom_level) + "newPoint4"

    connection = set_up_connection(hostname, port, user, password, database)

    # Plot one single tile
    # tms = morecantile.tms.get("WGS1984Quad")
    # tile = morecantile.Tile(x=134, y=106, z=10)
    # plot_tile(tile, column_names, connection)

    # Plot from mouse location to heatmap using matplotlib or folium
    # lat_long_query_result = lat_long_to_geo_df(mouse_lat, mouse_long, zoom_level, column_names, connection)
    # plot_heatmap(lat_long_query_result, heatmap_name)
    # draw_heatmap_folium(lat_long_query_result, radius, heatmap_name)

    # Plot from ROI to heatmaps using matplotlib
    # plot_all_tiles_within_ROI(west, east, south, north, zoom_level, column_names, connection)

    # Plot 9 tiles based on the central tile to check the border effect
    tms = morecantile.tms.get("WGS1984Quad")
    tile = morecantile.Tile(133, 106, 10)
    tile_name = str(tile)
    center_tile_to_9_tiles(tile, column_names, tile_name, connection)

    # Plot 9 tiles based on the mouse location to check the border effect
    # mouse_to_9_tiles(mouse_lat, mouse_long, zoom_level, column_names, connection)
    

    

 # %%

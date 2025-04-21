# %%
import os, uuid, shutil
import numpy as np

import pandas as pd
import geopandas as gpd
import rasterio, cv2
from rasterio.transform import from_origin
from rasterio.features import rasterize
from skimage.morphology import skeletonize

import morecantile
from shapely.geometry import box, LineString, Polygon
from pyproj import Transformer

import requests
from requests.adapters import HTTPAdapter, Retry
import multiprocessing
from joblib import Parallel, delayed

N_PIXELS = 256
SIZE_PIXEL = 1000
SESSION = requests.Session()
__retry = Retry(
    total=5, backoff_factor=0.5, 
    status_forcelist=[429, 500, 502, 503, 504]
)
__adapter = HTTPAdapter(max_retries=__retry)
SESSION.mount('https://', __adapter)
__lock = multiprocessing.Manager().Lock()

_stats_names = ['count', 'area', 'diameter', 'dia_min', 'dia_max', 'dia_median',
                'perimeter', 'width', 'LCP_count', 'iwn_len']

def gen_pixel_bounds(cell_bounds):
    minx, miny, maxx, maxy = cell_bounds
    x = range(round(minx), round(maxx), SIZE_PIXEL)
    y = range(round(miny), round(maxy), SIZE_PIXEL)
    assert len(x) == N_PIXELS and len(y) == N_PIXELS, "Invalid cell bounds."

    return [(i, j, i + SIZE_PIXEL, j + SIZE_PIXEL) for j in y for i in x]

def get_intersected_tiles(bounds, tms, zoom=15):
    # Define the transformer from EPSG:3413 to EPSG:4326
    transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326")

    # Create a polygon from the bounding box
    polygon = box(*bounds)

    # Define the interval for sampling points
    interval = SIZE_PIXEL // 5  
    
    # Sample points along the edges of the polygon
    points = []
    for i in range(len(polygon.exterior.coords) - 1):
        line = LineString([polygon.exterior.coords[i], polygon.exterior.coords[i + 1]])
        num_points = int(line.length // interval)
        points.extend([line.interpolate(float(j) / num_points, normalized=True) for j in range(num_points + 1)])

    # Convert the points to a list of coordinates
    coords = [(point.x, point.y) for point in points]

    # Transform the coordinates
    coords = [transformer.transform(x, y) for x, y in coords]
    coords = [(y, x) for x, y in coords]

    pixel_bbox = gpd.GeoDataFrame(geometry=[Polygon(coords)])

    # get all tiles that intersect with the polygon
    tiles = list(tms.tiles(*pixel_bbox.total_bounds, zooms=zoom))
    bbox_func = lambda x: box(x.left, x.bottom, x.right, x.top)
    filtered_tiles = [tile for tile in tiles if pixel_bbox.intersects(bbox_func(tms.bounds(tile))).any()]
    # print(len(filtered_tiles))

    return filtered_tiles

def download_tile(tile, download_root='downloads'):
    url = f'https://arcticdata.io/data/10.18739/A2KW57K57/iwp_geopackage_high/WGS1984Quad/{tile.z}/{tile.x}/{tile.y}.gpkg'
    download_path = os.path.join(download_root, f'{tile.z}_{tile.x}_{tile.y}.gpkg')
    
    try:
        # Send a HEAD request to check if the URL is available
        response = requests.head(url)
        if response.status_code == 200:
            if int(response.headers['Content-Length']) > 1024**3:
                raise RuntimeError('oversized source file detected.')
            # URL is available, proceed to download
            response = SESSION.get(url)
            with open(download_path, 'wb') as f:
                f.write(response.content)
            return True
    except requests.RequestException as e:
        print(f"Error checking URL: {url}, Error: {e}", file=open('dl_err.log', 'a'))

    return False

def tiles_from_local(tile, local_dir='data', working_dir='downloads'):
    local_path = os.path.join(local_dir, f'{tile.z}/{tile.x}/{tile.y}.gpkg')
    download_path = os.path.join(working_dir, f'{tile.z}_{tile.x}_{tile.y}.gpkg')
    if os.path.exists(local_path):
        os.link(local_path, download_path)
        return True
    return False

def IWP_skelenize(geoms, bounds, size=SIZE_PIXEL, kernel_size=11):
    xmin, _, _, ymax = bounds
    IWP_raster = rasterize(
        [(geom, 1) for geom in geoms],
        out_shape=(size, size),
        transform=from_origin(xmin, ymax, 1, 1),
        fill=0, dtype=np.uint8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    IWP_dilated = cv2.dilate(IWP_raster, kernel, iterations=1)
    difference_raster = IWP_dilated - IWP_raster
    IWP_skeleton = skeletonize(difference_raster)

    # from PIL import Image
    # im = Image.fromarray(IWP_skeleton.astype(np.uint8) * 255)
    # im.save('skeleton.png')
    # print(bounds)
    
    # with rasterio.open(
    #     'skeleton.tif', 'w',
    #     driver='GTiff',
    #     height=size,
    #     width=size,
    #     count=1,
    #     dtype=np.uint8,
    #     crs='EPSG:3413',
    #     transform=from_origin(xmin, ymax, 1, 1),
    # ) as dst:
    #     dst.write(IWP_skeleton, 1)

    return IWP_skeleton

def data_analyse(tiles, bounds, file_root='downloads', crs='EPSG:3413'):
    gdf = gpd.GeoDataFrame()
    for tile in tiles:
        tile_path = os.path.join(file_root, f'{tile.z}_{tile.x}_{tile.y}.gpkg')
        if os.path.getsize(tile_path) > 1024**3:
            with __lock:
                _gdf = gpd.read_file(tile_path)
                dedup_gdf = _gdf[_gdf['staging_duplicated'] == False]
                del _gdf
        else:
            _gdf = gpd.read_file(tile_path)
            dedup_gdf = _gdf[_gdf['staging_duplicated'] == False]
        gdf = pd.concat([gdf, dedup_gdf], ignore_index=True)

    # fileter the data based on the column 'centroidX' and 'centroidY' to get the data within the polygon
    inbox_gdf = gdf[
        gdf['CentroidX'].between(bounds[0], bounds[2]) & 
        gdf['CentroidY'].between(bounds[1], bounds[3])
    ]

    # get the skeleton of the IWP
    inbox_gdf = inbox_gdf.to_crs(crs)
    IWP_skeleton = IWP_skelenize(inbox_gdf['geometry'], bounds)
    
    # print(inbox_gdf.columns)
    stats = [
        len(inbox_gdf),
        inbox_gdf['Area'].sum(),
        inbox_gdf['Length'].sum(),
        inbox_gdf['Length'].min(),
        inbox_gdf['Length'].max(),
        inbox_gdf['Length'].median(),
        inbox_gdf['Perimeter'].sum(),
        inbox_gdf['Width'].sum(),
        (inbox_gdf['Class'].astype(int) == 1).sum(),
        IWP_skeleton.sum()
    ]

    return stats

def process_pixel(index, pixel_bounds, tms, zoom=15, remote=True):
    # generate uuid for the process
    process_uuid = str(uuid.uuid4())
    dl_root = '.' + process_uuid
    os.makedirs(dl_root)

    stats = [0.] * len(_stats_names)
    try:
        # Get the intersected tiles
        tiles = get_intersected_tiles(pixel_bounds, tms, zoom)
        # Download the tiles
        if remote:
            downloaded_tiles = [tile for tile in tiles 
                                if download_tile(tile, download_root=dl_root)]
        else:
            downloaded_tiles = [tile for tile in tiles 
                                if tiles_from_local(tile, local_dir='data', working_dir=dl_root)]
        # Analyse the data
        if len(downloaded_tiles) > 0:
            stats = data_analyse(downloaded_tiles, pixel_bounds, file_root=dl_root)
    except RuntimeError:
        stats = [-99.] * len(_stats_names)
    except Exception as e:
        print(f"Error processing pixel {index}, Error: {e}", file=open('proc_err.log', 'a'))
    finally:
        # Clean up
        shutil.rmtree(dl_root)
        return stats

def save_matrix_as_geotiff(matrix, cell_bounds, output_path, crs='EPSG:3413'):
    height, width = matrix.shape
    xmin, _, _, ymax = cell_bounds

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=matrix.dtype,
        crs=crs,
        transform=from_origin(round(xmin), round(ymax), SIZE_PIXEL, SIZE_PIXEL),
    ) as dst:
        dst.write(np.flipud(matrix), 1)

def main(cell_index, n_workers=20):
    from tqdm import tqdm

    tms = morecantile.tms.get("WGS1984Quad")
    grid = gpd.read_file("grid_230.geojson")
    cell = grid.loc[cell_index - 1, "geometry"]

    pixel_bounds = gen_pixel_bounds(cell.bounds)
    # ymin, xmin = -421744, -1811198
    # pixel_bounds = [[xmin, ymin, xmin + SIZE_PIXEL, ymin + SIZE_PIXEL]]

    # Process the pixel in parallel
    mapper = Parallel(n_jobs=n_workers)
    process = delayed(process_pixel)
    results = mapper(process(i, pixel, tms) for i, pixel in enumerate(tqdm(pixel_bounds)))

    # Save the results as a GeoTIFF
    cell_name = f"cell_{cell_index}"
    os.makedirs(cell_name, exist_ok=True)
    np_results = np.array(results).reshape(N_PIXELS, N_PIXELS, len(_stats_names))
    for i, name in enumerate(_stats_names):
        # print (np_results[..., i])
        save_matrix_as_geotiff(np_results[..., i], cell.bounds, f'{cell_name}/{cell_name}_{name}.tif')

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(int(sys.argv[1]), n_workers=12)
    else:
        exit("Please provide the cell index.")
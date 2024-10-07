# AI-Driven-trend-and-Pattern-analysis-tool
Spatial analysis tool for visualizing ice wedge polygons from PDG

## 1. Objectives
Ice Wedge is a crack in the ground formed by a narrow or thin piece of ice that measures up to 3–4 meters in length at ground level. Ice wedge is detected and stored into vector polygon (either geopackage or shapefile) in an arctic projection EPSG:3413.

The goal of this project is to visualize and analysis the polygons, and generate data products such as polygon count maps, heatmaps, or coverage ratio maps.

This tutorial will walk us through how to download these Ice Wedge Polygon (IWP), how to process the data (such as project conversion), and how to query data from database, and finally generate spatial analysis maps based on queried data. 

### 1.1. Create count maps for ice wedge polygons at tile level
The first goal is to create count maps of these IWP at tile level. 

In web mapping, a tiling system divides maps into small, square images (tiles) that can be loaded dynamically as needed by the user. Each tile represents a small portion of the larger map at a specific zoom level. The XYZ tiling system is commonly used in web mapping (e.g., Google Maps, OpenStreetMap). The map is divided into tiles based on x, y coordinates and a zoom level (z). The more you zoom in, the higher the zoom level and the more tiles are required to cover the same geographic area.

The count of IWP within each of pixels in a tile is calculated and visualized to show the distribution.  

![image](https://github.com/user-attachments/assets/088dced2-880f-40f9-94ff-165be14f9fec)


### 1.2. Create coverage ratio map for ice wedge polygon with given pixel resolution

The second goal is to create coverage ratio map of these IWP given a specific pixel resolution.

In coverage ratio map, we are not going to construct based on tiling system. Instead, we will take users’ ROI as a whole mapping region with a given resolution (such as 1km2, 2km2).

The coverage ratio map will calculate the coverage ratio of polygons within each pixel.

![image](https://github.com/user-attachments/assets/6530f21f-861b-4878-bdbd-82f50ce893da)



## 2. Data
### 2.1. Ice wedge polygon visualization portal
[Ice Wedge Polygon](https://arcticdata.io/catalog/portals/permafrost?lt=69.79173661318887&ln=-150.89470753194112&ht=1836228.7523939316&hd=1.0791169026958185&p=-89.55059855299719&r=0&el=iwp-coverage%2Ciwp%2Cosm)

### 2.2. Ice wedge polygon shapefile
[IWP shapefile (Alaska, Canada. Russia)](https://arcticdata.io/data/10.18739/A2KW57K57/iwp_shapefile_detections/high/alaska/)

### 2.3. Ice wedge polygon geopackage

[Introduction to geopackage data which deduplicated the footprints in shapefile data](https://github.com/PermafrostDiscoveryGateway/viz-staging/blob/main/docs/footprints.md)

[IWP polygon Geopackages (Canada, Alaska, Russia)](https://arcticdata.io/data/10.18739/A2KW57K57/iwp_geopackage_high/WGS1984Quad/)


## 3.	CICI Remote server connection
### 3.1.	Connect to cicilab remote server on local host device
In terminal on local host device, type:
ssh username@cici.lab.asu.edu
and then type password to connect to the server

### 3.2.	(Preferable) Download geopackages data into remote server
Since the geopackage folder is too large to load all the folders under it, we can check all the directories/data under the folder first: 

wget -r -np -nH --cut-dirs=3 -R '\?C=' https://arcticdata.io/data/10.18739/A2KW57K57/iwp_geopackage_high/WGS1984Quad/15

### 3.3.	(Optional, if shapefile is needed) Download shp data into remote server
Create folder in remote server to store data, type:
mkdir data/download
Go into the data folder to download data, type:
cd data/download

wget -r -np -nH --cut-dirs=3 -R '\?C=' https://arcticdata.io/data/10.18739/A2KW57K57/iwp_shapefile_detections/high/alaska
The shapefile will be downloaded into this folder.


## 4. Environment setup
### 4.1. Anaconda installtioin
Open the terminal in server, and following the [installation tutorial](https://docs.anaconda.com/anaconda/install/linux/) to install Anaconda.

### 4.2. Python packages installation
Open server terminal, to install the following python packages through conda for further analysis: rasterio, psycopg2, pyproj, shapely, pandas, geopandas, matplotlib, numpy, morecantile.


## 5. Database



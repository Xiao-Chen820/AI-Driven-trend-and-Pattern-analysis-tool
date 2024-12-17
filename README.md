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


### 1.2. Create various statistics maps for ice wedge polygons

The second goal is to create various statistics (coverage/width/length/perimeter sum, count) maps of these IWP given a specific pixel resolution (such as 1km).

The data process pipeline is at a grid level. However, the statistics are calculated at a pixel level. For example,

![22231734394918_ pic](https://github.com/user-attachments/assets/d2b6cc39-aa2c-4ae5-adbd-de04e28cdc20)



## 2. Data
### 2.1. Ice wedge polygon visualization portal
[Ice Wedge Polygon](https://arcticdata.io/catalog/portals/permafrost?lt=69.79173661318887&ln=-150.89470753194112&ht=1836228.7523939316&hd=1.0791169026958185&p=-89.55059855299719&r=0&el=iwp-coverage%2Ciwp%2Cosm)

### 2.2. Ice wedge polygon shapefile
[IWP shapefile (Alaska, Canada. Russia)](https://arcticdata.io/data/10.18739/A2KW57K57/iwp_shapefile_detections/high/alaska/)

### 2.3. Ice wedge polygon geopackage

[Introduction to geopackage data which deduplicated the footprints in shapefile data](https://github.com/PermafrostDiscoveryGateway/viz-staging/blob/main/docs/footprints.md)

[IWP polygon Geopackages (Canada, Alaska, Russia)](https://arcticdata.io/data/10.18739/A2KW57K57/iwp_geopackage_high/WGS1984Quad/)


## 3. Environment setup
### 3.1. Anaconda installtioin
Open the terminal in server, and following the [installation tutorial](https://docs.anaconda.com/anaconda/install/linux/) to install Anaconda.

### 3.2. Python packages installation
Open server terminal, to install the following python packages through conda for further analysis: rasterio, psycopg2, pyproj, shapely, pandas, geopandas, matplotlib, numpy, morecantile.

### 3.3.	PostgreSQL installed in Docker 
Since our PostgreSQL and PostGIS are installed in the docker in remote server, we need to specify the data location where the docker is listening to, so that data can be imported from the directory we just created in remote server to PostgreSQL database.
We might want our data stored within the docker to have higher I/O performance and better capability of data migration in the future, we will use Volumes instead of Bind mounts to mount the directory in docker container.
(To know more about docker storage, please check out Docker Storage)
 
We can mount the data directory when a docker container is created, but after it’s created, we can’t change the directory location to be mounted. 
The existing docker container installed with PostgreSQL has mounted a data directory somewhere else. In order to mount the new data folder we just created to store our project data, we have to create a new docker container and mount the data folder when it is being created. To do that, we can create a new docker container from the original docker container where PostgreSQL is installed within based on its docker image. (To know more about docker container and image, please check out this.)

1)	Check the original docker container name and its image ID that we want to create based on:
To list all the information about the existing docker containers:

`docker ps -a`
 

To list all the information about the existing docker images: 

`docker images`
 

Then we will get the following information that we need:

Docker name: ontop-docker_postgis_1
Docker image name: postgis/postgis
The image ID: 2027022e7219

2)	Stop the original docker container so that we can create a new docker container based on its image.

`docker stop ontop-docker_postgis_1`

3)	Create a new docker image from the existing docker image.

`docker commit 2027022e7219 postgis/postgis`

4)	Check if the image we just created exists

`docker images`

5)	Check the data directory in the server that we want to map into the docker, for example:

`/home/xchen/data`

6)	Specify a new folder that we want to use inside the docker to store the data, for example:

`/home/xchen/data`

7)	Create a new container from image and run with local folder mounted. Here we also want to adjust the default container shared memory to a large size, so we can have more space to do with large database. Here we set it to 100GB.

`docker run -it --shm-size=100gb -v local_folder:folder_in_container --name new_container_name -p container_port:host_port -t image_name:tag`

To be specific, e.g.,

`docker run -it --shm-size=100gb -v /home/xchen/data:/home/xchen/data --name postgis_xchen -p 5432:5432 -t postgis2:latest`


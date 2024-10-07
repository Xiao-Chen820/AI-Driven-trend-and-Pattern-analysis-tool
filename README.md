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

`ssh username@cici.lab.asu.edu`

and then type password to connect to the server

### 3.2.	(Preferable) Download geopackages data into remote server
Since the geopackage folder is too large to load all the folders under it, we can check all the directories/data under the folder first: 


`wget -r -np -nH --cut-dirs=3 -R '\?C=' https://arcticdata.io/data/10.18739/A2KW57K57/iwp_geopackage_high/WGS1984Quad/15`


### 3.3.	(Optional, if shapefile is needed) Download shp data into remote server
Create folder in remote server to store data, type:

`mkdir data/download`

Go into the data folder to download data, type:

`cd data/download`

`wget -r -np -nH --cut-dirs=3 -R '\?C=' https://arcticdata.io/data/10.18739/A2KW57K57/iwp_shapefile_detections/high/alaska`


The shapefile will be downloaded into this folder.


## 4. Environment setup
### 4.1. Anaconda installtioin
Open the terminal in server, and following the [installation tutorial](https://docs.anaconda.com/anaconda/install/linux/) to install Anaconda.

### 4.2. Python packages installation
Open server terminal, to install the following python packages through conda for further analysis: rasterio, psycopg2, pyproj, shapely, pandas, geopandas, matplotlib, numpy, morecantile.


## 5. Database
### 5.1.	PostgreSQL installed in Docker 
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

### 5.2.	Database connection
Here, we have three ways to connect to PostgreSQL. The first one is using psql (a terminal-based front-end to PostgreSQL), the second one is using Pgadmin 4 (a a graphical user interface administration tool to manage PostgreSQL), and the third one is using pyscopg2 (a python package to connect to PostgreSQL)

1)	Testing connection with psql
After ssh connect to remote server, type:

`psql -h cici.lab.asu.edu -p 5432 -U postgres -d postgres`

Then we will be required to type the user password.

2)	Testing connection with Pgadmin 4

a)	Connect to ASU VPN

b)	Right click on the Server to register a server

In Gerneral tab, 

Name: Alaska

In Connection tab, 

Host name/address: cici.lab.asu.edu

Port: 5432

Maintenance database: postgres

Username: (type username of database here)

Password: (type user password of database here)

In SSH Tunnel tab, turn on Use SSH Tunneling

Tunnel host: cici.lab.asu.edu

Tunnel port: 22

Username: (Type your username of remote server here)

Password: (Type your password of remote server here)


c)	Data tables will be imported under the schemas -> public -> table

3)	Connect to database using psycopg2
psycopg2 package can be used to connect to the database:

```
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

conn = set_up_connection(hostname, port, user, password, database)
In the future steps, geopandas will be used to read shapefile from database after connection is built up successfully:
def sql_to_geodataframe(query, conn):
   pd.set_option('display.max_colwidth', None)
   geo_df = gpd.read_postgis(query, conn, geom_col='geom_centroid', crs="3413")
   return geo_df
```


### 5.3.	Import data from remote server to PostgreSQL in docker
As introduced in the previous section 5.2., psql can be used to connect to database. After connect to the database, we can load data downloaded in the remote server to the database using shp2pgsql.
shp2pgsql is an SQL script which help us to import shapefile into database via terminal (To know more about shp2pgsql, please check out this).
 
The following tutorial will walk you through the key points in the code. The complete code is concluded the last step in this section.

1)	Since the PostgreSQL is installed in a docker container, if we want to start to use PostgreSQL, we should start with executing the docker container.

`docker exec postgis_xchen`

2)	The above code will be followed by the shp2pgsql command to create a new table.

`docker exec postgis_xchen shp2pgsql -S -s 3413 $shpPath $tableName`

3)	Create a PGPASSFILE file. When we import hundreds of shapefile into database, we will be authenticated for hundreds of times. To avoid the manual password typing, we can create a PGPASSFILE file named .pgpass in the server user folder (which is /home/xchen in this case). In the .pgpass file, type:

`cici.lab.asu.edu:5432:your_database_name:your_database_username:your_database_password`

e.g., 

`cici.lab.asu.edu:5432:postgres:postgres:your_database_password`

4)	Import the project data (mentioned in section 2.3.) from server using shp2pgsql, to the PostgreSQL using psql

`docker exec postgis_xchen shp2pgsql -S -s 3413 $shpPath $tableName | PGPASSFILE='/home/xchen/.pgpass' psql -h localhost -U postgres -d postgres -q`

5)	The import process may last for couple hours, so we can also hang the program in the backend of server to avoid the issue caused by laptop shutdown, for example, using nohup:

`nohup docker exec postgis_xchen shp2pgsql -S -s 3413 -a $shpPath $tableName | PGPASSFILE='/home/xchen/.pgpass' psql -h localhost -U postgres -d postgres -q`

6)	As we have multiple shapefile under the alaska folder, we can iterate all the shapefile using shell script. We will create a shell script file (such as shpImport.sh), and the entire code looks like the following:

```
#!/bin/bash

dir="/home/xchen/data/download_alaska/iwp_shapefile_detections/high/alaska/*/*/*.shp"
flag=0
echo "flag is: $flag"
start=$(date +%s)
for shpPath in $dir
do
    echo "Go to folder: $shpPath"
    shpName="$(shp=${shpPath##*/}; echo ${shp%.*})"
    echo $shpName
    tableName='alaska_all_3413'
    export PGPASSWORD='shirly'
    echo $PGPASSWORD
    if [ $flag = 0 ]
    then 
        echo "flag before creating is: $flag"
        echo "-------------------Creating A Table!------------------"
        docker exec postgis_xchen shp2pgsql -S -s 3413 $shpPath $tableName | PGPASSFILE='/home/xchen/.pgpass' psql -h localhost -U postgres -d postgres -q
        flag=$(( $flag + 1 ))
        echo "flag after creating is: $flag"
    else 
        echo "flag before inserting is: $flag"
        echo "-------------------Inserting into the existing table!-----------------"
        docker exec postgis_xchen shp2pgsql -S -s 3413 -a $shpPath $tableName | PGPASSFILE='/home/xchen/.pgpass' psql -h localhost -U postgres -d postgres -q
    fi
done

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
echo "----------------------Table combined!-------------------"
```

### 5.4.	Check the newly added table in pgadmin 4
After the shapefile is loaded into PostgreSQL, we can check it out in pgadmin 4. The total number of records in Alaska region are 119,124,921.

![image](https://github.com/user-attachments/assets/54bc27f7-1552-4bc6-a442-5c3eee0d8d72)



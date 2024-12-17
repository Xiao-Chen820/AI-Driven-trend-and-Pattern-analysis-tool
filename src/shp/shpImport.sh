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

# docker exec postgis_xchen shp2pgsql /home/xchen/data/download_alaska/iwp_shapefile_detections/high/alaska/146_157_iwp/GE01_20110826213903_10504100013F3800_11AUG26213903-M1BS-054019163020_01_P001_u16rf3413_pansh/demo1 firstDemo | psql -h localhost -U postgres -d test_xchen3
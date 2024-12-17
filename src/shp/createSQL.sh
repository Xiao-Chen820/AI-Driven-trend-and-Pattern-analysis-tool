#!/bin/bash

# Variables
SQL_FILE="/home/xchen/data/SQLcode/vacuum.sql"
CONTAINER_NAME="postgis_xchen"
DB_NAME="postgres"
DB_USER="postgres"
PGPASSFILE="/home/xchen/.pgpass"  # Path to your .pgpass file
LOG_FILE="/home/xchen/data/SQLcode/vacuum.log"

# docker exec -i postgis_xchen PGPASSFILE='/home/xchen/.pgpass' psql -h localhost -U postgres -d postgres -q -f Convert2Points.sql
nohup docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -f $SQL_FILE > $LOG_FILE 2>&1 &
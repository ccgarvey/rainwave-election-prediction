#!/bin/bash
# This script is used to parse the data collected for a given station.
# to be invoked with parse_station_data PATH STATION_NAME OUT_FILE_PATH
for file in $1/$2*.data; do
    echo "Parsing \"${file}\"..."
    python3 parse_data_restricted.py "${file}" $3
done

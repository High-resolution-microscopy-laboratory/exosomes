#!/usr/bin/env bash

COMMAND=$1
ROOT=$2
IN_DIR=$3
OUT_DIR=$4

if [[ "$#" -eq 4 ]]; then
    OUT="-o data/${OUT_DIR}"
elif [[ "$#" -eq 3 ]]; then
    OUT=""
else
    echo "Invalid number of arguments"
fi

docker container run -v ${ROOT}:/app/data/ vesicles ${COMMAND} -i data/${IN_DIR} ${OUT}
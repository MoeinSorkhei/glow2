#!/usr/bin/env bash
# Code from: https://github.com/phillipi/pix2pix/blob/master/datasets/download_dataset.sh

DATASET_NAME=$1

if [[ $DATASET_NAME != "cityscapes" &&  $DATASET_NAME != "night2day" &&  $DATASET_NAME != "edges2handbags" && $DATASET_NAME != "edges2shoes" && $DATASET_NAME != "facades" && $DATASET_NAME != "maps" ]]; then
  echo "Available datasets are cityscapes, night2day, edges2handbags, edges2shoes, facades, maps"
  exit 1
fi

echo "Specified [$DATASET_NAME]"

URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$DATASET_NAME.tar.gz
# TAR_FILE=./data/$DATASET_NAME/$DATASET_NAME.tar.gz  # e.g. data/maps/maps.tar.gz
# TARGET_DIR=$DATASET_NAME/  # e.g. data/maps
TAR_FILE=/Midgard/Data/moein/$DATASET_NAME/$DATASET_NAME.tar.gz  # e.g. /Midgard/Data/moein/maps/maps.tar.gz
TARGET_DIR=/Midgard/Data/moein/$DATASET_NAME/  # e.g. /Midgard/Data/moein/maps/

mkdir -p $TARGET_DIR
wget -N $URL -O $TAR_FILE
tar -zxf $TAR_FILE -C $TARGET_DIR
rm $TAR_FILE

# tar options:
# -z: for gzip, -x: extract, -f: denotes the archive file, -C: specifies the directory for extraction


# examples usages:
# bash scripts/others/download_data.sh maps
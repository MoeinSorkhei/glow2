#!/usr/bin/env bash

# ============== downloading the trained Caffe model
if [[ "$1" == "rs_city_samples_midgard" ]]; then
  URL=http://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/fcn-8s-cityscapes/fcn-8s-cityscapes.caffemodel
  OUTPUT_FILE=evaluation/third_party/caffemodel/fcn-8s-cityscapes.caffemodel
  wget -N $URL -O $OUTPUT_FILE
fi


# ============== downloading cityscapes...


if [[ "$1" == "prep_trans_data" ]]; then
  URL=http://transattr.cs.brown.edu/files/aligned_images.tar
  OUTPUT_FILE=/Midgard/Data/moein/transient/aligned_images.tar

  URL2=http://transattr.cs.brown.edu/files/annotations.tar
  OUTPUT_FILE2=/Midgard/Data/moein/transient/annotations.tar

  # download
  echo "Downloading" $URL "to" $OUTPUT_FILE
  mkdir -p "/Midgard/Data/moein/transient"
  wget -N $URL -O $OUTPUT_FILE

  echo "Downloading" $URL2 "to" $OUTPUT_FILE2
  wget -N $URL2 -O $OUTPUT_FILE2

  # extract
  echo "Extracting..."
  # mkdir -p "imageAlignedLD"
  tar -xf aligned_images.tar # -C imageAlignedLD  -x: extract, -f: file name, -C output dir
  tar -xf annotations.tar

  # remove tar fiels
  rm aligned_images.tar
  rm annotations.tar
fi


if [[ "$1" == "move_trans" ]]; then
  # salloc --constrain="gondor|shire|rivendell|khazadum|belegost"
  cp -a /Midgard/Data/moein/transient/ /local_storage/datasets/moein/
fi

# salloc --constrain="gondor|khazadum"
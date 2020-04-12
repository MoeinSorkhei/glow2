#!/bin/bash
# ============== downloading the trained Caffe model
if [[ "$1" == "rs_city_samples_midgard" ]]; then
  URL=http://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/fcn-8s-cityscapes/fcn-8s-cityscapes.caffemodel
  OUTPUT_FILE=evaluation/third_party/caffemodel/fcn-8s-cityscapes.caffemodel
  wget -N $URL -O $OUTPUT_FILE
fi


# ============== downloading cityscapes...

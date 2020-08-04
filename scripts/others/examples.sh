#!/usr/bin/env bash


# ======= move boundary maps
# train
rsync -Pav /Midgard/Data/moein/cityscapes/boundaries/gtFine/train/ \
           /local_storage/datasets/moein/cityscapes/gtFine_trainvaltest/gtFine/train/

# val
rsync -Pav /Midgard/Data/moein/cityscapes/boundaries/gtFine/val/ \
           /local_storage/datasets/moein/cityscapes/gtFine_trainvaltest/gtFine/val/


import torch

# this device is accessible in all the functions in this file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fixed conditioning
segmentations = [
    "/jena/jena_000078_000019_gtFine_color.png",
    "/jena/jena_000067_000019_gtFine_color.png",
    "/jena/jena_000011_000019_gtFine_color.png",
    "/jena/jena_000066_000019_gtFine_color.png",
    "/strasbourg/strasbourg_000001_061472_gtFine_color.png"
]

real_imgs = [
    "/jena/jena_000078_000019_leftImg8bit.png",
    "/jena/jena_000067_000019_leftImg8bit.png",
    "/jena/jena_000011_000019_leftImg8bit.png",
    "/jena/jena_000066_000019_leftImg8bit.png",
    "/strasbourg/strasbourg_000001_061472_leftImg8bit.png"
]

# adding the base path
base_seg_path = "/local_storage/datasets/moein/cityscapes/gtFine_trainvaltest/gtFine/train"
base_real_path = "/local_storage/datasets/moein/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train"

desired_segmentations = [base_seg_path + segmentations[i] for i in range(len(segmentations))]
desired_real_imgs = [base_real_path + real_imgs[i] for i in range(len(real_imgs))]

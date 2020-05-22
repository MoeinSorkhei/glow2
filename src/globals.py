import torch
from torchvision import transforms

# this device is accessible in all the functions in this file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# global image transforms
city_transforms = transforms.Compose([transforms.Resize([256, 256]),
                                      transforms.ToTensor()])

# base paths (now only from train set)
base_seg_path = "/local_storage/datasets/moein/cityscapes/gtFine_trainvaltest/gtFine/train"
base_real_path = "/local_storage/datasets/moein/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train"

# =====================================  for training
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

# for fixed conditioning during training
desired_segmentations = [base_seg_path + segmentations[i] for i in range(len(segmentations))]
desired_real_imgs = [base_real_path + real_imgs[i] for i in range(len(real_imgs))]


# =====================================  for experiments
random_sampling_reals = [    # whose segmentations will be used for random sampling
    "/jena/jena_000078_000019_leftImg8bit.png",
    "/jena/jena_000067_000019_leftImg8bit.png",
    "/jena/jena_000011_000019_leftImg8bit.png",
    "/jena/jena_000066_000019_leftImg8bit.png",
    "/strasbourg/strasbourg_000001_061472_leftImg8bit.png"
]

# for random sampling experiments
sampling_real_imgs = [base_real_path + random_sampling_reals[i] for i in range(len(random_sampling_reals))]


new_conds = []  # the real images whose segmentations will be used as new conditions
new_cond_reals = {
    'orig_img0': base_real_path + "/aachen/aachen_000002_000019_leftImg8bit.png",
    'cond_imgs0': [base_real_path + "/jena/jena_000011_000019_leftImg8bit.png"],

    'orig_img1': base_real_path + "/aachen/aachen_000010_000019_leftImg8bit.png",
    'cond_imgs1': [base_real_path + "/jena/jena_000011_000019_leftImg8bit.png"],

    'orig_img': base_real_path + "/jena/jena_000066_000019_leftImg8bit.png",
    'cond_imgs': [
        # base_real_path + "/jena/jena_000011_000019_leftImg8bit.png"
        base_real_path + "/aachen/aachen_000010_000019_leftImg8bit.png",
    ],

    'orig_img3': base_real_path + "/aachen/aachen_000028_000019_leftImg8bit.png",
    'cond_imgs3': [base_real_path + "/jena/jena_000011_000019_leftImg8bit.png"],

    'orig_img4': base_real_path + "/jena/jena_000011_000019_leftImg8bit.png",
    'cond_imgs4': [
        base_real_path + "/aachen/aachen_000002_000019_leftImg8bit.png",
        base_real_path + "/aachen/aachen_000010_000019_leftImg8bit.png",
        base_real_path + "/jena/jena_000011_000019_leftImg8bit.png",
        base_real_path + "/jena/jena_000066_000019_leftImg8bit.png",
        base_real_path + "/aachen/aachen_000011_000019_leftImg8bit.png",
        base_real_path + "/aachen/aachen_000012_000019_leftImg8bit.png",
        base_real_path + "/aachen/aachen_000012_000019_leftImg8bit.png",
        base_real_path + "/aachen/aachen_000013_000019_leftImg8bit.png",
        base_real_path + "/aachen/aachen_000015_000019_leftImg8bit.png",
        base_real_path + "/aachen/aachen_000025_000019_leftImg8bit.png",
        base_real_path + "/aachen/aachen_000027_000019_leftImg8bit.png",
        base_real_path + "/aachen/aachen_000028_000019_leftImg8bit.png",

    ]
}

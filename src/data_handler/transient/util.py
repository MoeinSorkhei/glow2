import csv
import itertools
from torchvision import transforms


fixed_conds = {
    'daylight2night': ['00000325/10.jpg', '00000325/24.jpg', '00000325/42.jpg'],

    # warm2cold and sunny2clouds
    'warm2cold+sunny2clouds': ['00000325/10.jpg', '00000325/24.jpg', '00000325/42.jpg'],
    'spring2autumn+summer2winter+dry2rain': ['00000064/131.jpg', '00000064/142.jpg', '00000064/86.jpg']
}

attributes = [
    'dirty',
    'daylight',
    'night',
    'sunrisesunset',
    'dawndusk',
    'sunny',
    'clouds',
    'fog',
    'storm',
    'snow',
    'warm',
    'cold',
    'busy',
    'beautiful'
    'flowers',
    'spring',
    'summer',
    'autumn',
    'winter',
    'glowing',
    'colorful',
    'dull',
    'rugged',
    'midday',
    'dark',
    'bright',
    'dry',
    'moist',
    'windy',
    'rain',
    'ice',
    'cluttered',
    'soothing',
    'stressful',
    'exciting',
    'sentimental',
    'mysterious',
    'boring',
    'gloomy',
    'lush',
]

# validation cameras
# val_cameras = ['00023947', '90000002', '90000003']
val_cameras = ['90000002', '90000003']

# 11 test cameras
test_cameras = ['90000004', '90000005', '90000006', '90000007', '90000008', '90000009',
                '90000010', '90000011', '90000012', '90000013', '90000014']


def init_transformer(img_size):
    trans = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    return trans


def read_annotations(annotations_path):
    with open(annotations_path) as annotations:
        reader = csv.reader(annotations, delimiter='\t')
        annotations = [line for line in reader]  # convert iterable to list

    all_cameras = set([line[0].split('/')[0] for line in annotations])
    return annotations, all_cameras


def unique_names(lines):
    names = [line[0] for line in lines]
    return set(names)


def retrieve_imgs_with_attribute(annotations, attr_index):
    imgs = []
    for line in annotations:
        if float(line[attr_index].split(',')[0]) > 0.8:  # split the confidence from score
            imgs.append(line)
    return imgs


def create_multi_pairs(annotations_path, directions, split):
    all_pairs = []
    for direction in directions.split('+'):
        pairs = create_pairs_for_direction(annotations_path, direction, split)
        all_pairs.extend(pairs)

    print(f'In [create_multi_pairs]: total pairs for split {split} is : {len(all_pairs):,}')
    return all_pairs


def create_pairs_for_direction(annotations_path, direction, split):
    left_attr, right_attr = direction.split('2')
    annotations, _ = read_annotations(annotations_path)

    # +1 since in the annotations file index 0 is for the file name, so attr indices start from 1
    left_ind, right_ind = attributes.index(left_attr) + 1, attributes.index(right_attr) + 1
    all_uniques = []  # unique cameras that have this attribute
    all_names = []  # all the images from all the cameras that have the attribute

    # for ind in [ind_spring, ind_winter, ind_day, ind_night, ind_warm, inc_cold]:
    for attr_ind in [left_ind, right_ind]:
        imgs = retrieve_imgs_with_attribute(annotations, attr_ind)
        names = [img[0] for img in imgs]  # take the name from the annotation line, e.g 00000064/29.jpg
        uniques = list(set([name.split('/')[0] for name in names]))

        all_names.append(names)
        all_uniques.append(uniques)

    # all camera names that have images of both attributes, e.g. 00000064
    equals = list(set(all_uniques[0]) & set(all_uniques[1]))

    # all the images from all the cameras with the attribute
    all_left_attr = all_names[0]
    all_right_attr = all_names[1]

    # extracting image pairs from the cameras that hve both attributes
    left_attr_imgs = []
    right_attr_imgs = []
    all_pairs = []

    # create (day, night) pair per camera
    for camera in equals:
        if split == 'val' and camera not in val_cameras:
            continue
        if split == 'test' and camera not in test_cameras:
            continue
        if split == 'train' and (camera in val_cameras or camera in test_cameras):
            continue

        camera_left = [left for left in all_left_attr if camera in left]
        camera_right = [right for right in all_right_attr if camera in right]

        # all possible combinations of (left, right) images
        pairs = list(itertools.product(camera_left, camera_right))

        left_attr_imgs.extend(camera_left)
        right_attr_imgs.extend(camera_right)
        all_pairs.extend(pairs)

    print(f'In [create_pairs]: total pairs for split {split} is : {len(all_pairs):,}')
    return all_pairs



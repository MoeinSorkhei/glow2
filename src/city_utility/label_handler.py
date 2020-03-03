"""
I will work with the IDs themselves rather than trainID. So there are 34 IDs in total excluding 'license plate'.
"""
import json

from .labels import *


def info_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    objects = data['objects']  # returns list of ('label', 'polygon')
    # print(f'In [json_to_labels]: read the json file with {len(objects)} objects in it.')

    all_ids = []  # all the object IDs in the image - might have repeats for instances of an object
    id_repeats = [0] * 34

    for obj in objects:
        obj_name = assureSingleInstanceName(obj['label'])  # e.g. cargroup => car
        obj_label = name2label[obj_name]  # Label contains ID, trainID, category and so on
        obj_id = obj_label.id
        obj_poly = obj['polygon']

        if obj_id != -1:
            id_repeats[obj_id] += 1

            # if obj_id == 26 and obj_label.name == 'car':
            #    print('found car')

        # print(type(obj), type(obj_name))
        # print(obj_name)
        # print(obj_label)
        # print(obj_id)
        # print(obj_poly, '\n\n')
        # input()

    # print('In [json_to_labels]: id_repeats=', id_repeats)
    # print('Total count:', sum(id_repeats))
    return {'id_repeats': id_repeats}


if __name__ == '__main__':
    file = '../../data/cityscapes/gtFine_trainvaltest/gtFine/train/aachen/aachen_000000_000019_gtFine_polygons.json'
    info_from_json(file)


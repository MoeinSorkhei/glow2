import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def make_grid(array):
    # assumes shape  (5, H, W, C)
    h, w, c = array.shape[1:]
    grid = np.transpose(array, (1, 0, 2, 3))  # (H, 5, W, C)
    grid = np.reshape(grid, (h, 5 * w, c))  # make a 2D grid
    return grid


def read_and_preprocess_image(path):
    image = Image.open(path).resize((256, 256))  # read image and resize
    image_array = np.array(image)[:, :, :3] / 255  # remove alpha channel
    processed_tf = tf.cast(image_array, tf.float32)
    return processed_tf


def write_tfrecords(tfrecord_file, seg_img_paths, real_img_paths):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _serialize_example(seg_image, real_image):
        feature = {
            'segment_image': _bytes_feature(seg_image),
            'real_image': _bytes_feature(real_image),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for i in range(len(seg_img_paths)):
            seg_path = seg_img_paths[i]
            seg_img_array = read_and_preprocess_image(seg_path)

            real_path = real_img_paths[i]
            real_img_array = read_and_preprocess_image(real_path)

            seg_img_bytes = tf.io.serialize_tensor(seg_img_array)
            real_img_bytes = tf.io.serialize_tensor(real_img_array)

            example = _serialize_example(seg_img_bytes, real_img_bytes)
            writer.write(example)

            if i % 100 == 0:
                print(f'In [write_tfrecords]: wrote tfrecord for the {i}th record')


def read_tfrecords(tfrecord_file, dataset_name, direction, batch_size, is_training):
    def reshape_and_expand(left, right):
        left = tf.reshape(left, [256, 256, 3])
        left = tf.expand_dims(left, axis=0)  # for depth

        right = tf.reshape(right, [256, 256, 3])
        right = tf.expand_dims(right, axis=0)
        return left, right

    def _read_maps_serialized_example(serialized_example):
        feature_description = {
            'the_map': tf.io.FixedLenFeature((), tf.string),
            'the_photo': tf.io.FixedLenFeature((), tf.string),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)
        the_map = tf.io.parse_tensor(example['the_map'], out_type=tf.float32)
        the_photo = tf.io.parse_tensor(example['the_photo'], out_type=tf.float32)

        the_map, the_photo = reshape_and_expand(the_map, the_photo)
        if direction == 'map2photo':
            return the_map, the_photo
        return the_photo, the_map

    def _read_city_serialized_example(serialized_example):
        feature_description = {
            'segment_image': tf.io.FixedLenFeature((), tf.string),
            'real_image': tf.io.FixedLenFeature((), tf.string),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)
        seg_image = tf.io.parse_tensor(example['segment_image'], out_type=tf.float32)
        real_image = tf.io.parse_tensor(example['real_image'], out_type=tf.float32)

        seg_image, real_image = reshape_and_expand(seg_image, real_image)
        if direction == 'label2photo':
            return seg_image, real_image  # returned by the iterator
        return real_image, seg_image

    dataset = tf.data.TFRecordDataset(tfrecord_file)
    tfrecords_len = sum(1 for _ in tf.python_io.tf_record_iterator(tfrecord_file))  # number of tfrecord elements
    print('In [read_tfrecords]: Created dataset with tfrecords of len:', tfrecords_len)

    # create iterators
    if dataset_name == 'cityscapes':
        dataset = dataset.map(_read_city_serialized_example)
    elif dataset_name == 'maps':
        dataset = dataset.map(_read_maps_serialized_example)
    else:
        raise NotImplementedError

    if is_training:
        dataset = dataset.shuffle(buffer_size=tfrecords_len)
        print(f'In [read_tfrecords]: shuffling dataset with buffer_size {tfrecords_len}: done')

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    # dataset = dataset.prefetch(buffer_size=1)
    itr = dataset.make_one_shot_iterator()
    return itr


def write_data_for_tf(tfrecord_file, gt_folder):
    tf.enable_eager_execution()  # needed for writing tfrecords

    segment_imgs = read_image_ids(data_folder=gt_folder, dataset_name='cityscapes_segmentation')
    real_imgs = [real_name_from_segment_name(segment_name) for segment_name in segment_imgs]
    write_tfrecords(tfrecord_file, segment_imgs, real_imgs)


# ============== helper functions
def visualize(mode, item=None, dataset=None, n_items=None):
    if mode == 'dataset':
        for i, data in enumerate(dataset.take(n_items)):
            img = tf.keras.preprocessing.image.array_to_img(tf.squeeze(data[0], axis=0))
            img.show()
            img = tf.keras.preprocessing.image.array_to_img(tf.squeeze(data[1], axis=0))
            img.show()
    else:
        segment = tf.keras.preprocessing.image.array_to_img(tf.squeeze(item[0], axis=0))
        real = tf.keras.preprocessing.image.array_to_img(tf.squeeze(item[1], axis=0))
        segment.show()
        real.show()


def real_name_from_segment_name(segment_name):
    return segment_name.replace('/gtFine_trainvaltest/', '/leftImg8bit_trainvaltest/') \
                       .replace('/gtFine/', '/leftImg8bit/')\
                       .replace('_gtFine_color.png', '_leftImg8bit.png')


def read_image_ids(data_folder, dataset_name):  # duplicated code from the helper package
    """
    It reads all the image names (id's) in the given data_folder, and returns the image names needed according to the
    given dataset_name.

    :param data_folder: to folder to read the images from. NOTE: This function expects the data_folder to exist in the
    'data' directory.

    :param dataset_name: the name of the dataset (is useful when there are extra unwanted images in data_folder, such as
    reading the segmentations)

    :return: the list of the image names.
    """
    img_ids = []
    if dataset_name == 'cityscapes_segmentation':
        suffix = '_color.png'
    elif dataset_name == 'cityscapes_leftImg8bit':
        suffix = '_leftImg8bit.png'
    else:
        raise NotImplementedError('In [read_image_ids] of Dataset: the wanted dataset is not implemented yet')

    # all the files in all the subdirectories
    for city_name, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(suffix):  # read all the images in the folder with the desired suffix
                img_ids.append(os.path.join(city_name, file))

    # print(f'In [read_image_ids]: found {len(img_ids)} images')
    return img_ids


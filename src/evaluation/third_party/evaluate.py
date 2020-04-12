"""
Adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/eval_cityscapes/evaluate.py
"""
import os
os.environ['GLOG_minloglevel'] = '2'  # level 2: warnings - suppressing caffe verbose prints
import caffe
import argparse
import numpy as np
import scipy.misc
from PIL import Image
from .util import segrun, fast_hist, get_scores
from .cityscapes import cityscapes
import sys


def evaluate(data_folder, paths, split='val', save_output_images=True, gpu_id=0):
    output_dir = paths['eval_results']  # Where to save the evaluation result
    result_dir = paths['resized_path']  # Path to the (resized) generated images to be evaluated
    cityscapes_dir = data_folder
    # IMPORTANT ASSUMPTION: the program is run from the main.py module
    caffemodel_dir = 'evaluation/third_party/caffemodel'

    print(f'In [evaluate]: images will be read from: "{result_dir}"')
    print(f'In [evaluate]: results will be saved to: "{output_dir}"')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if save_output_images > 0:
        output_image_dir = output_dir + '/image_outputs'
        print(f'In [evaluate]: output images will be saved to: "{output_image_dir}"')
        if not os.path.isdir(output_image_dir):
            os.makedirs(output_image_dir)

    CS = cityscapes(cityscapes_dir)
    n_cl = len(CS.classes)
    label_frames = CS.list_label_frames(split)
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    # caffe.set_mode_cpu()
    net = caffe.Net(caffemodel_dir + '/deploy.prototxt',
                    caffemodel_dir + '/fcn-8s-cityscapes.caffemodel',
                    caffe.TEST)
    os.environ['GLOG_minloglevel'] = '1'  # level 1: info - back to normal
    print('In [evaluate]: Caffe model setup: done')

    hist_perframe = np.zeros((n_cl, n_cl))
    for i, idx in enumerate(label_frames):
        if i % 50 == 0:
            print('Evaluating: %d/%d' % (i, len(label_frames)))

        city = idx.split('_')[0]
        # idx is city_shot_frame
        label = CS.load_label(split, city, idx)  # label shape: (1, 1024, 2048)
        im_file = result_dir + '/' + idx + '_leftImg8bit.png'

        im = np.array(Image.open(im_file))  # im shape: (1024, 2048, 3)
        im = scipy.misc.imresize(im, (label.shape[1], label.shape[2]))  # assumption: scipy=1.0.0 installed

        out = segrun(net, CS.preprocess(im))
        hist_perframe += fast_hist(label.flatten(), out.flatten(), n_cl)
        if save_output_images > 0:
            label_im = CS.palette(label)
            pred_im = CS.palette(out)

            # assumption: scipy=1.0.0 installed
            scipy.misc.imsave(output_image_dir + '/' + str(i) + '_pred.jpg', pred_im)
            scipy.misc.imsave(output_image_dir + '/' + str(i) + '_gt.jpg', label_im)
            scipy.misc.imsave(output_image_dir + '/' + str(i) + '_input.jpg', im)
            # mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)

            # print(f'In [evaluate]: saved prediction to: "{output_image_dir + "/" + str(i) + "_pred.jpg"}"')
            # print(f'So far histogram: \n'
            #      f'mean_pixel_acc = {mean_pixel_acc} \n'
            #      f'mean_class_acc = {mean_class_acc} \n'
            #      f'mean_class_iou = {mean_class_iou} \n')

    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    print(f'Histogram: \n'
          f'mean_pixel_acc = {mean_pixel_acc} \n'
          f'mean_class_acc = {mean_class_acc} \n'
          f'mean_class_iou = {mean_class_iou} \n')

    with open(output_dir + '/evaluation_results.txt', 'w') as f:
        f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc)
        f.write('Mean class accuracy: %f\n' % mean_class_acc)
        f.write('Mean class IoU: %f\n' % mean_class_iou)
        f.write('************ Per class numbers below ************\n')
        for i, cl in enumerate(CS.classes):
            while len(cl) < 15:
                cl = cl + ' '
            f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))


def args_to_dict(args):
    """
    Creates a dictionary from the arguments, used in tracking the experiment using comet.
    :param args: program arguments.
    :return: a dictionary.
    """
    return {'cityscapes_dir': args.cityscapes_dir,
            'result_dir': args.result_dir,
            'output_dir': args.output_dir}


def print_pred(pred_im):
    """
    Printing different labels with their frequencies in the prediction (used to visually see which different classes
    are predicted by the classifier).
    :param pred_im: -
    :return: -
    """
    print('In [print_pred]: printing the prediction')
    flat = pred_im.flatten().astype(np.int64)
    counts = np.bincount(flat)
    ii = np.nonzero(counts)[0]
    print(list(zip(ii, counts[ii])))

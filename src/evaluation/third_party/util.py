# The following code is modified from https://github.com/shelhamer/clockwork-fcn
import numpy as np
import os


def get_out_scoremap(net):
    return net.blobs['score'].data[0].argmax(axis=0).astype(np.uint8)


def feed_net(net, in_):
    """
    Load prepared input into net.
    """
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_


def segrun(net, in_):
    feed_net(net, in_)
    net.forward()
    return get_out_scoremap(net)


def fast_hist(a, b, n):
    # this first line takes out negative elements and those classes that are not used for evaluation (trainIds -1 and 255)
    k = np.where((a >= 0) & (a < n))[0]
    bc = np.bincount(n * a[k].astype(int) + b[k], minlength=n**2)
    if len(bc) != n**2:
        # ignore this example if dimension mismatch
        return 0
    return bc.reshape(n, n)


def get_scores(hist):
    # Mean pixel accuracy
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)

    # Per class accuracy
    cl_acc = np.diag(hist) / (hist.sum(1) + 1e-12)

    # Per class IoU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)

    return acc, np.nanmean(cl_acc), np.nanmean(iu), cl_acc, iu


def get_score_and_print(hist_perframe, city_classes, verbose=False, save_path=None, sampling_round=None):
    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    print('Mean pixel accuracy: %f\n' % mean_pixel_acc)
    print('Mean class accuracy: %f\n' % mean_class_acc)
    print('Mean class IoU: %f\n' % mean_class_iou)

    if verbose:
        print('************ Per class numbers below ************\n')
        for i, cl in enumerate(city_classes):
            while len(cl) < 15:
                cl = cl + ' '  # adding spaces
            print('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))

    if save_path:
        # file = indexed_eval_file(output_dir=save_path)
        file = os.path.join(save_path, f'evaluation_results_{sampling_round}.txt')
        with open(file, 'w') as f:
            f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc,)
            f.write('Mean class accuracy: %f\n' % mean_class_acc)
            f.write('Mean class IoU: %f\n' % mean_class_iou)
            f.write('************ Per class numbers below ************\n')
            for i, cl in enumerate(city_classes):
                while len(cl) < 15:
                    cl = cl + ' '
                f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))
        print(f'Results written to: "{os.path.abspath(file)}"')


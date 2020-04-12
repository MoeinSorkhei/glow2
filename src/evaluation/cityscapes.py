import torch

from . import third_party
import helper
import experiments


def evaluate_city(args, params):
    if args.gt:  # for ground-truth images
        paths = {'resized_path': '/Midgard/home/sorkhei/glow2/data/cityscapes/resized/val',
                 'eval_results': '/Midgard/home/sorkhei/glow2/gt_eval_results'}
    else:
        paths = helper.compute_paths(args, params)

    third_party.evaluate(data_folder=params['data_folder']['base'], paths=paths, split='val')
    print(f'In [evaluate_city]: evaluation done')


def eval_complete(args, params, device):
    for temp in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f'In [eval_complete]: evaluating for temperature = {temp}')
        params['temperature'] = temp

        # inference
        experiments.infer_on_validation_set(args, params, device)
        torch.cuda.empty_cache()  # very important
        print('In [eval_complete]: inference done')

        # resize
        helper.resize_for_fcn(args, params)
        print('In [eval_complete]: resize done')

        # evaluate
        evaluate_city(args, params)
        print(f'In [eval_complete]: evaluating for temperature = {temp}: done')

    torch.cuda.empty_cache()  # very important
    print('In [eval_complete]: all done')

import torch

from . import third_party
import helper
import experiments
import data_handler
from train import calc_val_loss
from globals import device
import helper
import models


def evaluate_city(args, params):
    if args.gt:  # for ground-truth images
        paths = {'resized_path': '/Midgard/home/sorkhei/glow2/data/cityscapes/resized/val',
                 'eval_results': '/Midgard/home/sorkhei/glow2/gt_eval_results'}
    else:
        paths = helper.compute_paths(args, params)

    output_dir = paths['eval_results']
    # no resize for 256x256, so we read from validation path directly
    result_dir = paths['resized_path'] if params['img_size'] != [256, 256] else paths['val_path']
    # third_party.evaluate(data_folder=params['data_folder']['base'], paths=paths, split='val')
    third_party.evaluate(data_folder=params['data_folder']['base'],
                         output_dir=output_dir,
                         result_dir=result_dir,
                         split='val')
    print(f'In [evaluate_city]: evaluation done')


def eval_complete(args, params, device):
    for temp in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f'In [eval_complete]: evaluating for temperature = {temp}')
        params['temperature'] = temp

        # inference
        experiments.infer_on_validation_set(args, params, device)
        torch.cuda.empty_cache()  # very important
        print('In [eval_complete]: inference done \n')

        # resize
        if params['img_size'] == [256, 256]:
            print(f'In [eval_complete]: no resize since the image size already is {params["img_size"]} \n')
        else:
            helper.resize_for_fcn(args, params)
            print('In [eval_complete]: resize done \n')

        # evaluate
        evaluate_city(args, params)
        print(f'In [eval_complete]: evaluating for temperature = {temp}: done \n')

    torch.cuda.empty_cache()  # very important
    print('In [eval_complete]: all done \n')


def compute_val_bpd(args, params):
    loader_params = {'batch_size': params['batch_size'], 'shuffle': False, 'num_workers': 0}
    _, val_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                  image_size=(params['img_size']),
                                                  remove_alpha=True,  # removing the alpha channel
                                                  loader_params=loader_params)

    '''checkpoints_path = helper.compute_paths(args, params)['checkpoints_path']
    optim_step = args.last_optim_step
    model = models.init_model(args, params, device, run_mode='infer')
    model, _, _ = helper.load_checkpoint(checkpoints_path, optim_step, model, None, device)'''
    model = models.init_and_load(args, params, run_mode='infer')

    mean, std = calc_val_loss(args, params, device, model, val_loader)
    print(f'In [compute_val_bpd]: mean = {mean} - std = {std}')

import data_handler
import models
from globals import device
from trainer import calc_val_loss
from .fcn import *
import experiments


def eval_city_with_all_temps(args, params):
    """
    Performs steps needed for evaluation with all the temperatures.
    :param args:
    :param params:
    :return:
    """
    for temp in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f'In [eval_city_with_all_temps]: evaluating for temperature = {temp}')
        params['temperature'] = temp

        # inference
        experiments.infer_on_validation_set(args, params)
        torch.cuda.empty_cache()  # very important
        print('In [eval_city_with_all_temps]: inference done \n')

        # current implementation fo label2photo requires that generated images are resized and save to the folder
        if args.direction == 'label2photo':
            # resize if not 256x256
            if params['img_size'] == [256, 256]:
                print(f'In [eval_city_with_all_temps]: no resize since the image size already is {params["img_size"]} \n')
            else:
                helper.resize_for_fcn(args, params)
                print('In [eval_city_with_all_temps]: resize done \n')

        # evaluate
        eval_city_with_temp(args, params)
        print(f'In [eval_city_with_all_temps]: evaluating for temperature = {temp}: done \n')

    torch.cuda.empty_cache()  # very important
    print('In [eval_city_with_all_temps]: all done \n')


def eval_city_with_temp(args, params):
    """
    Evaluate the generated validation images wit the given temperature. This function is called from
    eval_city_with_all_temps function that tries different temperatures.
    If this function is called individually, the it used the temperature specified in params to find the correct path for
    the images that are to be evaluated.

    :param args:
    :param params:
    :return:
    """
    if args.direction == 'label2photo':
        if args.gt:  # for ground-truth images (photo2label only)
            paths = {'resized_path': '/Midgard/home/sorkhei/glow2/data/cityscapes/resized/val',
                     'eval_results': '/Midgard/home/sorkhei/glow2/gt_eval_results'}
        else:
            paths = helper.compute_paths(args, params)

        output_dir = paths['eval_results']
        # no resize for 256x256, so we read from validation path directly
        result_dir = paths['resized_path'] if params['img_size'] != [256, 256] else paths['val_path']

        evaluate_real_imgs_with_temp(data_folder=params['data_folder']['base'],
                                     output_dir=output_dir,
                                     result_dir=result_dir,
                                     split='val')

    else:
        evaluate_segmentations_with_temp(args, params)
    print(f'In [eval_city_with_temp]: evaluation done')


def compute_val_bpd(args, params):
    loader_params = {'batch_size': params['batch_size'], 'shuffle': False, 'num_workers': 0}
    _, val_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                  image_size=(params['img_size']),
                                                  remove_alpha=True,  # removing the alpha channel
                                                  loader_params=loader_params)

    model = models.init_and_load(args, params, run_mode='infer')

    mean, std = calc_val_loss(args, params, model, val_loader)
    print(f'In [compute_val_bpd]: mean = {mean} - std = {std}')

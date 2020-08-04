import data_handler
import models
from trainer import calc_val_loss
from .fcn import *
from .ssim import *
import experiments
import helper


def evaluate_fcn(args, params):
    assert args.dataset == 'cityscapes' and params['img_size'] == [256, 256]  # not supported otherwise for now

    # specify the paths
    if args.direction == 'label2photo' and args.gt:  # evaluating ground-truth images
        all_paths = {'resized_path': '/Midgard/home/sorkhei/glow2/data/cityscapes/resized/val',
                     'eval_results': '/Midgard/home/sorkhei/glow2/gt_eval_results'}
    else:
        all_paths = helper.compute_paths(args, params)

    syn_dir = extend_val_path(all_paths['val_path'], args.sampling_round)  # val_imgs_1, val_imgs_2, ...
    eval_dir = all_paths['eval_path']
    print(f'In [evaluate_city_fcn]: results will be read from: "{syn_dir}"')

    # evaluation
    if args.direction == 'label2photo':
        eval_real_imgs_with_temp(base_data_folder=params['data_folder']['base'],
                                 synthesized_dir=syn_dir,
                                 save_dir=eval_dir,
                                 sampling_round=args.sampling_round)
    else:
        eval_segmentations_with_temp(synthesized_dir=syn_dir,
                                     reference_dir=params['data_folder']['segment'],
                                     base_data_folder=params['data_folder']['base'],
                                     save_dir=eval_dir,
                                     sampling_round=args.sampling_round)
    print(f'In [eval_city_with_temp]: evaluation done')


def eval_ssim(args, params):
    assert args.dataset == 'cityscapes'  # not implemented otherwise
    if args.direction == 'label2photo':
        ref_dir = os.path.join(params['data_folder']['real'], 'val')
    else:
        ref_dir = os.path.join(params['data_folder']['segment'], 'val')
    syn_dir = helper.extend_val_path(helper.compute_paths(args, params)['val_path'], args.sampling_round)

    # get the score
    ssim_score = compute_ssim(syn_dir, ref_dir)

    ssim_file = os.path.join(helper.compute_paths(args, params)['eval_path'], 'ssim.txt')
    with open(ssim_file, 'a') as file:
        file.write(f'Round {args.sampling_round}: {ssim_score}\n')

    print(f'In [eval_ssim]: writing ssim score of {ssim_score} to: "{ssim_file}"')
    print(f'In [eval_ssim]: all done.')


def eval_all_rounds(args, params, eval_mode):
    assert eval_mode == 'fcn' or eval_mode == 'ssim'
    for sampling_round in [1, 2, 3]:
        args.sampling_round = sampling_round
        print(f'In [eval_all_rounds]: doing for sampling round {sampling_round}')

        if eval_mode == 'ssim':
            eval_ssim(args, params)
        else:
            evaluate_fcn(args, params)
        print(f'In [eval_all_rounds]: done for sampling round {sampling_round}\n\n')


# fcn with all temperature
def eval_fcn_all_temps(args, params):
    # might be buggy
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
                print(
                    f'In [eval_city_with_all_temps]: no resize since the image size already is {params["img_size"]} \n')
            else:
                helper.resize_for_fcn(args, params)
                print('In [eval_city_with_all_temps]: resize done \n')

        # evaluate
        # infer_and_evaluate_c_flow(args, params)
        evaluate_fcn(args, params)  # NOT TESTED
        print(f'In [eval_city_with_all_temps]: evaluating for temperature = {temp}: done \n')

    torch.cuda.empty_cache()  # very important
    print('In [eval_city_with_all_temps]: all done \n')


def compute_val_bpd(args, params):
    loader_params = {'batch_size': params['batch_size'], 'shuffle': False, 'num_workers': 0}
    _, val_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                  image_size=(params['img_size']),
                                                  remove_alpha=True,  # removing the alpha channel
                                                  loader_params=loader_params)

    model = models.init_and_load(args, params, run_mode='infer')

    mean, std = calc_val_loss(args, params, model, val_loader)
    print(f'In [compute_val_bpd]: mean = {mean} - std = {std}')

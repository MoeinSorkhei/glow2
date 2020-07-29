import os

import torch
from torchvision import utils

import data_handler
import helper
import models
from .cityscapes import *
from globals import device, new_cond_reals
from helper import load_checkpoint


def infer_on_validation_set(args, params):
    """
    The model name and paths should be equivalent to the name used in the resize_for_fcn function
    in evaluation.third_party.prepare.py module.
    :param args:
    :param params:
    :return:
    """
    with torch.no_grad():
        paths = helper.compute_paths(args, params)
        checkpt_path, val_path = paths['checkpoints_path'], paths['val_path']
        val_path = helper.extend_val_path(val_path, args.sampling_round)

        print(f'In [infer_on_validation_set]:\n====== checkpt_path: "{checkpt_path}" \n====== val_path: "{val_path}" \n')
        helper.make_dir_if_not_exists(val_path)

        # init and load model
        model = models.init_and_load(args, params, run_mode='infer')
        print(f'In [infer_on_validation_set]: init model and load checkpoint: done')

        # validation loader
        batch_size = params['batch_size']
        loader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 0}
        _, val_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                      image_size=(params['img_size']),
                                                      remove_alpha=True,  # removing the alpha channel
                                                      loader_params=loader_params)
        print('In [infer_on_validation_set]: loaded val_loader of len:', len(val_loader))

        # inference on validation set
        print('In [infer_on_validation_set]: starting inference on validation set')
        for i_batch, batch in enumerate(val_loader):
            img_batch = batch['real'].to(device)
            segment_batch = batch['segment'].to(device)
            boundary_batch = batch['boundary'].to(device) if args.cond_mode == 'segment_boundary' else None
            real_paths = batch['real_path']  # list: used to save samples with the same name as original images
            seg_paths = batch['segment_path']

            # create reverse conditions and samples based on args.direction
            rev_cond = models.batch2revcond(args, img_batch, segment_batch, boundary_batch)
            samples = models.take_samples(args, params, model, rev_cond)

            # save inferred images separately
            paths_list = real_paths if args.direction == 'label2photo' else seg_paths
            save_one_by_one(samples, paths_list, val_path)

            if i_batch > 0 and i_batch % 20 == 0:
                print(f'In [infer_on_validation_set]: done for the {i_batch}th batch out of {len(val_loader)} batches')
        print(f'In [infer_on_validation_set]: all done. Inferred images could be found at: {val_path} \n')


def sample_trained_c_flow(args, params):
    checkpt_pth = params['checkpoints_path']['real'][args.cond_mode][args.model]
    # could also use the init_and_load function
    model = models.init_model(args, params, run_mode='infer')
    model, _, _ = load_checkpoint(checkpt_pth, args.last_optim_step, model, None, False)  # loading the model

    if args.conditional:
        sample_c_flow_conditional(args, params, model)

    if args.syn_segs:
        syn_new_segmentations(args, params, model)


def sample_with_new_condition(args, params):
    """
    In this function, temperature has no effect here as we have no random sampling.
    :param args:
    :param params:
    :return:
    """
    model = models.init_and_load(args, params, run_mode='infer')
    orig_real_name = new_cond_reals['orig_img']  # image paths of the original image (with the desired style)
    orig_pure_name = orig_real_name.split('/')[-1][:-len('_leftImg8bit.png')]
    print(f'In [sample_with_new_cond]: orig cond is: "{orig_pure_name}" \n')

    # ========= get the original segmentation and real image
    orig_seg_batch, _, orig_real_batch, orig_bmap_batch = \
        data_handler.create_cond(params, fixed_conds=[orig_real_name], save_path=None)  # (1, C, H, W)

    # make b_maps None is not needed
    if args.cond_mode == 'segment':
        orig_bmap_batch = None

    # ========= real_img_name: the real image whose segmentation which will be used for conditioning.
    for new_cond_name in new_cond_reals['cond_imgs']:
        new_cond_pure_name = new_cond_name.split('/')[-1][:-len('_leftImg8bit.png')]
        additional_info = {
            'orig_pure_name': orig_pure_name,  # original condition city name
            'new_cond_pure_name': new_cond_pure_name,   # new condition city name
            'exp_type': 'new_cond'
        }
        print(f'In [sample_with_new_cond]: doing for images: "{new_cond_pure_name}" {"=" * 50} \n')

        # ========= getting new segment cond and real image batch
        exp_path, new_rev_cond, new_real_batch = prep_for_sampling(args, params, new_cond_name, additional_info)

        # ========= new_cond segment and bmap batches
        new_seg_batch = new_rev_cond['segment']  # (1, C, H, W)
        new_bmap_batch = new_rev_cond['boundary']

        if args.cond_mode == 'segment':
            new_bmap_batch = None

        # ========= save new segmentation and the corresponding real imgs
        helper.make_dir_if_not_exists(exp_path)

        utils.save_image(new_seg_batch.clone(), f"{exp_path}/new_seg.png", nrow=1, padding=0)
        utils.save_image(new_real_batch.clone(), f"{exp_path}/new_real.png", nrow=1, padding=0)

        # ========= save the original segmentation and real image
        utils.save_image(orig_seg_batch.clone(), f"{exp_path}/orig_seg.png", nrow=1, padding=0)
        utils.save_image(orig_real_batch.clone(), f"{exp_path}/orig_real.png", nrow=1, padding=0)
        print(f'In [sample_with_new_cond]: saved original and new segmentation images')

        # =========== getting z_real corresponding to the desired style (x_b) from the orig real images
        left_glow_outs, right_glow_outs = model(x_a=orig_seg_batch, x_b=orig_real_batch, b_map=orig_bmap_batch)
        z_real = right_glow_outs['z_outs']  # desired style

        # =========== apply the new condition to the desired style
        new_real_syn = model.reverse(x_a=new_seg_batch,  # new segmentation (condition)
                                     b_map=new_bmap_batch,
                                     z_b_samples=z_real,  # desired style
                                     mode='new_condition')  # (1, C, H, W)
        utils.save_image(new_real_syn.clone(), f"{exp_path}/new_real_syn.png", nrow=1, padding=0)
        print(f'In [sample_with_new_cond]: save synthesized real image')

        # =========== save all images of combined in a grid
        all_together = torch.cat([orig_seg_batch, new_seg_batch, orig_real_batch, new_real_batch, new_real_syn], dim=0)
        utils.save_image(all_together.clone(), f"{exp_path}/all.png", nrow=2, padding=10)

        exp_pure_path = exp_path.split('/')[-1]
        all_together_path = os.path.split(exp_path)[0] + '/all'  # 'all' dir in the previous dir
        helper.make_dir_if_not_exists(all_together_path)

        utils.save_image(all_together.clone(), f"{all_together_path}/{exp_pure_path}.png", nrow=2, padding=10)
        print(f'In [sample_with_new_cond]: for images: "{new_cond_pure_name}": done {"=" * 50} \n')


def take_random_samples(args, params):
    """
    This function takes random samples from a model specified in the args. See scripts for example usage of this function.
    Uses the globals.py file for obtaining the conditions for sampling.
    :param args:
    :param params:
    :return:
    """
    model, reverse_cond = models.init_model(args, params, run_mode='infer')
    print(f'In [take_random_samples]: model init: done')

    # temperature specified
    if args.temp:
        if args.model == 'c_flow':
            sample_c_flow(args, params, model)

        elif args.model == 'c_glow':
            if args.direction == 'label2photo':
                rev_cond = reverse_cond['segment']
            else:
                raise NotImplementedError

            models.take_samples(args, params, model, reverse_cond=rev_cond)

        else:
            raise NotImplementedError
        print(f'In [take_random_samples]: temperature = {params["temperature"]}: done')

    # try different temperatures
    else:
        for temp in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            print(f'In [take_random_samples]: doing for temperature = {temp}')
            params['temperature'] = temp

            if args.model == 'c_flow':
                sample_c_flow(args, params, model)

            elif args.model == 'c_glow':
                models.take_samples(args, params, model, reverse_cond=reverse_cond)

            else:
                raise NotImplementedError

            print(f'In [take_random_samples]: temperature = {temp}: done \n')
    print(f'In [take_random_samples]: all done \n')


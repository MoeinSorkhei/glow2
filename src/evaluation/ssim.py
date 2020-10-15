from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

import data_handler, helper
from globals import city_transforms


def compute_ssim_old(args, params):
    """
    images should be np arrays of shape (H, W, C). This function should be called only after the inference has been
    done on validation set. This function computes SSIM for the temperature specified in params['temperature'].
    :return:
    """
    # =========== init validation loader
    loader_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
    _, val_loader = data_handler.init_city_loader(data_folder=params['data_folder'],
                                                  image_size=(params['img_size']),
                                                  remove_alpha=True,  # removing the alpha channel
                                                  loader_params=loader_params)
    print(f'In [compute_ssim]: init val data loader of len {len(val_loader)}: done \n')

    # ============= computing the validation path for generated images
    if params['img_size'] != [256, 256]:
        raise NotImplementedError('Now only supports 256x256 images')  # should use paths['resized_path'] probably?

    paths = helper.compute_paths(args, params)
    val_path = paths['val_path']
    print(f'In [compute_ssim]: val_path to read from: \n"{val_path}" \n')

    # ============= SSIM calculation for all the images in validation set
    ssim_vals = []
    for i_batch, batch in enumerate(val_loader):  # for every single image (batch size 1)
        # reading reference (ground truth) and generated image
        ref_img = batch['real'].cpu().data.squeeze(dim=0).permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        pure_name = batch['real_path'][0].split('/')[-1]  # the city name of the image

        # compute corresponding path in generated images and read the image
        gen_img_path = val_path + f'/{pure_name}'
        gen_img = city_transforms(Image.open(gen_img_path)).cpu().data.permute(1, 2, 0).numpy()

        # ============= computing SSIM
        pixel_range = 1  # with data range 1 since pixels can take a value between 0 to 1
        ssim_val = ssim(ref_img, gen_img, multichannel=True, data_range=pixel_range)  # expects (H, W, C) ordering
        ssim_vals.append(ssim_val)

        if i_batch % 100 == 0:
            print(f'In [compute_ssim]: evaluated {i_batch} images')

    # ssim_score = round(np.mean(ssim_vals), 2)
    ssim_score = np.mean(ssim_vals)
    print(f'In [compute_ssim]: ssim score on validation set: {ssim_score}')

    # ============= append result to ssim.txt
    eval_path_base = paths['eval_path_base']
    with open(f'{eval_path_base}/ssim.txt', 'a') as ssim_file:
        string = f'temp = {params["temperature"]}: {ssim_score} \n'
        ssim_file.write(string)
        print(f'In [compute_ssim]: ssim score appended to ssim.txt')


def compute_ssim(file_paths, ref_dir):
    ssim_vals = []

    for i, filepath in enumerate(file_paths):
        # if not filepath.endswith('_leftImg8bit.png'):  # only consider files that end like this
        #     pass
        syn_image = np.array(Image.open(filepath).resize((2048, 1024)))  # resize to original size, (H, W, C) order
        ref_file = helper.get_file_with_name(ref_dir, helper.pure_name(filepath))
        ref_image = np.array(Image.open(ref_file))[:, :, :3]  # 1024x2048 with int values, removed alpha

        # ssim needs images as float
        syn_image = helper.image_as_float(syn_image)
        ref_image = helper.image_as_float(ref_image)

        ssim_val = ssim(ref_image, syn_image, multichannel=True, data_range=1.)
        ssim_vals.append(ssim_val)

        if i % 50 == 0:
            print(f'Done for images {i}/{len(file_paths)}')

    ssim_score = np.mean(ssim_vals)
    return ssim_score


def compute_ssim_all(args, params):
    # temperature specified
    if args.temp:
        compute_ssim(args, params)  # with the already adjusted temp
        print(f'In [compute_ssim_all]: done for specified temperature: {params["temperature"]}')

    # try different temperatures
    else:
        for temp in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            print(f'In [compute_ssim_all]: doing for temperature: {temp}')
            params['temperature'] = temp
            compute_ssim(args, params)
            print(f'In [compute_ssim_all]: for temperature: {temp}: done \n')
    print(f'In [compute_ssim_all]: all done')

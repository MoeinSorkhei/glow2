# import torch  # ==> some imports induce errors which are seemingly due to conda environemnt
# import models
# import data_handler
# import train
# import experiments
# import helper
from helper import resize_imgs, read_params, resize_for_fcn
import evaluation
import helper
from torchvision import utils

import data_handler
import helper

import matplotlib.pyplot as plt
# import json

import torch
# import evaluation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import models

def test_actnorm(which_fn):
    act = model.ActNorm(in_channel=2)
    mu, std = act.loc + 1, act.scale

    tens = torch.zeros((1, 2, 1, 1)) + 5

    # print((mu + tens) * 2 * std)
    # act.forward(inp=torch.zeros(1, 2, 1, 1))

    if which_fn == 'initialize':
        # test initializes
        batch = torch.ones(2, 2, 2, 2)
        batch[0:, 0, :, :] = 0
        batch[1:, 0, :, :] = 1

        batch[0:, 1, :, :] = 1
        batch[1:, 1, :, :] = 2

        # batch[0:, 2, :, :] = 2
        # batch[1:, 2, :, :] = 3

        # print(batch[:, 2, :, :])
        act.initialize(batch)

        # print(act.scale.size())


def test_read_cityscapes():
    # reading segmentation from the Aachen city in the training set
    seg_folder = 'cityscapes/gtFine_trainvaltest/gtFine/train/aachen'
    dataset = data_handler.CityDataset(data_folder=seg_folder, mode='cityscapes_segmentation')
    print(len(dataset))


def test_read_mnist():
    # do not know why the folder in which test.py runs is the project folder
    # while the running folder for main.py is the 'src' folder
    mnist_folder = "data/mnist"
    experiments.visualize_mnist(mnist_folder, img_index=14)


def test_save_mnist():
    experiments.save_mnist_rgb('data/mnist', 'data/mnist/images')


def test_resume_train():
    # reading params from the json file
    with open('params.json', 'r') as f:
        parameters = json.load(f)['mnist']  # parameters related to the wanted dataset

    optim_step = 9000
    device = torch.device('cpu')
    train.resume_train(optim_step, parameters, device)


def test_label_mnist():
    label = 1
    h, w = 20, 20
    t = train.label_to_tensor(1, h, w)
    print(t)


def test_list(l):
    l2 = [l[i] for i in range(len(l))]
    l2.append('2')
    print('l2:', l2)
    print('l:', l)


def test_resample():
    # reading params from the json file
    with open('../params.json', 'r') as f:
        parameters = json.load(f)['mnist']  # parameters related to the wanted dataset

    experiments.resample_latent(img_index=7, resample_lst=['z1'], params=parameters, args=None, device=device)


def test_city(mode):
    data_path = "data/cityscapes/gtFine_trainvaltest/gtFine/train"
    dataset_name = 'cityscapes_segmentation'

    data_path2 = "data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train"
    dataset_name2 = 'cityscapes_leftImg8bit'

    data_folder = {'segment': data_path, 'real': data_path2}

    params = helper.read_params('params.json')['cityscapes']

    if mode == 'read_imgs':
        img_ids = data_handler.read_image_ids(data_path, dataset_name)
        print(f'{len(img_ids)} images found, some of them:')

        for i in range(10):
            print(img_ids[i])

    if mode == 'visualize':
        img_path = 'aachen/aachen_000000_000019_gtFine_color.png'
        img_path2 = 'aachen/aachen_000000_000019_leftImg8bit.png'
        experiments.visualize_img(img_path2, data_path2, dataset_name2, desired_size=(32, 64))

    if mode == 'train':
        params = helper.read_params('params.json')['cityscapes']
        img_size = (params['channels'], params['height'], params['width'])
        cond_orig_size = img_size

        cond_shapes = helper.calc_cond_shapes(cond_orig_size, params['channels'],
                                              (params['height'], params['width']),
                                               params['n_block'])
        print('cond shapes:', cond_shapes)

        glow = models.init_glow(params, cond_shapes)
        loader_prms = {'batch_size': 10, 'shuffle': True, 'num_workers': 0}
        loader = data_handler.init_city_loader(data_folder, (params['height'], params['width']),
                                               remove_alpha=True, loader_params=loader_prms)
        for i_batch, batch in enumerate(loader):
            # plt.imshow(batch['real'].squeeze(0).permute(1, 2, 0))

            # plt.imshow(batch['segment'].squeeze(0).permute(1, 2, 0))
            real = batch['real']
            segment = batch['segment']
            cond = ('city_segment', segment)

            _, _, z = glow(real, cond)
            print('forward done')

            z_rand = train.sample_z(helper.calc_z_shapes(params['channels'], (params['height'], params['width']),
                                                         params['n_block']), real.shape[0], 0.7, device)
            inp = glow.reverse(z, reconstruct=True, coupling_conds=cond)
            print('backward first done')

            inp_rand = glow.reverse(z_rand, reconstruct=False, coupling_conds=cond)
            print('backward second done')

    if mode == 'c_flow':
        two_glows = models.TwoGlows(params)
        seg_path = 'aachen/aachen_000000_000019_gtFine_color.png'
        real_path = 'aachen/aachen_000000_000019_leftImg8bit.png'

        dataset = data_handler.CityDataset(data_folder, img_size=params['img_size'], remove_alpha=True)
        seg = dataset[0]['segment'].unsqueeze(0)
        real = dataset[0]['real'].unsqueeze(0)
        seg_path = dataset[0]['segment_path']

        print('Segment path:', seg_path)
        print('cond shapes:', two_glows.cond_shapes)
        # input()

        # total_log_det, z_outs_left, flows_outs_left, z_outs_right, flows_outs_right = two_glows(seg, real)
        total_log_det, _, left_glow_outs, right_glow_outs = two_glows(seg, real)

        # for i in range(len(left_glow_outs['z_outs'])):
        #    left_glow_outs['z_outs'][i].unsqueeze(0)
        #    right_glow_outs['z_outs'][i].unsqueeze(0)

        '''print('left z_outs shapes')
        for z in left_glow_outs['z_outs']:
            print(z.shape)

        print('right z_outs shapes')
        for z in right_glow_outs['z_outs']:
            print(z.shape)'''

        with torch.no_grad():
            x_a, x_b = two_glows.reverse(left_glow_outs, right_glow_outs, mode='reconstruct')

        x_a = x_a.squeeze(0)  # removing batch dimension (only one image)
        x_b = x_b.squeeze(0)
        helper.show_images([x_a, x_b])

        print(torch.allclose(x_a, seg), torch.allclose(x_b, real))
        print(torch.mean(torch.abs(x_a - seg)))
        print(torch.mean(torch.abs(x_b - real)))


def test_eval():
    params = read_params('../params.json')['city_evaluation']
    evaluation.eval_city_with_temp(params)


def test_resize():
    # helper.resize_imgs(path_to_load='data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val',
    #                   path_to_save='data/cityscapes/resized/val')

    # resize_imgs(path_to_load='/local_storage/datasets/moein/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val',
    #                   path_to_save='/local_storage/datasets/moein/cityscapes/resized/val')

    resize_for_fcn()


def test_cn():
    import models
    import torch

    actnorm_cn = models.ActNormCN(in_channels=3, cond_h=16, cond_w=32, fc_out_shape=24)
    a = torch.randn(10, 3, 16, 32)
    out = actnorm_cn(a)
    print('out shape:', out.shape)


def test_edge():
    import helper
    a = torch.randn((1, 3, 4, 4)).to(device)
    edge = helper.get_edges(a)
    print(edge)


def test_pil():
    from PIL import Image
    import glob

    p = "/local_storage/datasets/moein/cityscapes/gtFine_trainvaltest/gtFine"
    # p = "/Midgard/Data/moein/cityscapes/boundaries/"
    files = glob.glob(p + '/**/*_boundary.png', recursive=True)
    print(f'Found {len(files)} boundary images recursively at: "{p}"')

    # pth = "/local_storage/datasets/moein/cityscapes/gtFine_trainvaltest/gtFine" \
    #      "/train/strasbourg/strasbourg_000001_005666_gtFine_boundary.png"
    # Image.open(pth)
    for f in files:
        # if f != '/local_storage/datasets/moein/cityscapes/gtFine_trainvaltest/gtFine/' \
        #        'train/strasbourg/strasbourg_000001_005666_gtFine_boundary.png':
        Image.open(f)
            # print('open file done')
    print('All files are OK.')


def test_recreate_bmap():
    boundary_path = '/local_storage/datasets/moein/cityscapes/gtFine_trainvaltest/gtFine/' \
                'train/strasbourg/strasbourg_000001_005666_gtFine_boundary.png'

    instance_path = '/local_storage/datasets/moein/cityscapes/gtFine_trainvaltest/gtFine/' \
                'train/strasbourg/strasbourg_000001_005666_gtFine_instanceIds.png'

    helper.recreate_boundary_map(instance_path, boundary_path, device)


def test_cond_net():
    import models
    inp, stride = torch.randn((10, 6, 64, 128)), 3
    # inp, stride = torch.randn((10, 12, 32, 64)), 3
    # inp, stride = torch.randn((10, 24, 16, 32)), 2
    # inp, stride = torch.randn((10, 48, 8, 16)), 1
    cond_net = models.WCondNet(inp[0].shape, conv_stride=stride)
    out = cond_net(inp[0].unsqueeze(0))


def test_cond_actnorm():
    import models
    inp, stride = torch.randn((10, 6, 64, 128)), 3
    # inp, stride = torch.randn((10, 12, 32, 64)), 3
    # inp, stride = torch.randn((10, 24, 16, 32)), 2
    # inp, stride = torch.randn((10, 48, 8, 16)), 1

    # inp = inp[0].unsqueeze(0)
    cond = torch.randn(inp.shape)

    cond_act_norm = models.ActNormConditional(inp[0].shape, conv_stride=stride)

    # cond_net = models.ActCondNet(inp[0].shape, conv_stride=stride)
    # out = cond_net(inp)  # output shape (B, 2, C)

    # s, t = out[:, 0, :].unsqueeze(2).unsqueeze(3), out[:, 1, :].unsqueeze(2).unsqueeze(3)
    # print('s, t shape: ', s.shape, t.shape)
    # print('s, t: ', s, t)

    # print('inp mean:', inp[:, 0].mean())
    # print('inp std:', inp[:, 0].std())
    # print('\n')

    # output = s * (inp + t)
    #for i in range(4):
    #    print(f'out mean for i={i}:', output[:, i, :, :].mean())
    #    print(f'out std for i={i}:', output[:, i, :, :].std())
    #    print()

    # print(s[0] == s[1])
    # print(t[2] == t[3])

    out, _ = cond_act_norm(inp, act_left_out=cond)
    inp_again = cond_act_norm.reverse(out, act_left_out=cond)

    # print(inp == inp_again)


def test_cond_coupling():
    inp = torch.randn((10, 6, 256, 256))
    cond = models.CouplingCondNet(inp.shape[1:])
    out = cond(inp)

    print(f'Input shape: {inp.shape}')
    print(f'Output shape: {out.shape}')


def test_ssim():
    pass


def test_ssim_from_skimage():
    """
    Code from https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt

    from skimage import data, img_as_float
    from skimage.metrics import structural_similarity as ssim

    img = img_as_float(data.camera())
    rows, cols = img.shape

    noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
    noise[np.random.random(size=noise.shape) > 0.5] *= -1

    def mse(x, y):
        return np.linalg.norm(x - y)

    img_noise = img + noise
    img_const = img + abs(noise)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    mse_none = mse(img, img)
    ssim_none = ssim(img, img, data_range=img.max() - img.min())

    mse_noise = mse(img, img_noise)
    ssim_noise = ssim(img, img_noise,
                      data_range=img_noise.max() - img_noise.min())

    mse_const = mse(img, img_const)
    ssim_const = ssim(img, img_const,
                      data_range=img_const.max() - img_const.min())

    print(f'const image range: {img_const.max() - img_const.min()} \n'
          f'noise image range min: {img_noise.min()} - max: {img_noise.max()}: {img_noise.max() - img_noise.min()} \n'
          f'orignal img range: {img.max() - img.min()}')

    label = 'MSE: {:.2f}, SSIM: {:.2f}'

    ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[0].set_xlabel(label.format(mse_none, ssim_none))
    ax[0].set_title('Original image')

    ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[1].set_xlabel(label.format(mse_noise, ssim_noise))
    ax[1].set_title('Image with noise')

    ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[2].set_xlabel(label.format(mse_const, ssim_const))
    ax[2].set_title('Image plus constant')

    plt.tight_layout()
    # plt.show()


def eval_seg():
    import evaluation
    generated = torch.zeros((3, 3, 2))

    generated[0, 0, 0] = 126
    generated[1, 0, 0] = 64
    generated[2, 0, 0] = 128

    generated[0, 0, 1] = 126
    generated[1, 0, 1] = 64
    generated[2, 0, 1] = 128

    generated[0, 1, 0] = 242
    generated[1, 1, 0] = 35
    generated[2, 1, 0] = 232

    generated[0, 1, 1] = 242
    generated[1, 1, 1] = 35
    generated[2, 1, 1] = 232

    generated[0, 2, 0] = 1  # (0, 0, 230)
    generated[1, 2, 0] = 1
    generated[2, 2, 0] = 234

    generated[0, 2, 1] = 118  # (119, 11, 32)
    generated[1, 2, 1] = 10
    generated[2, 2, 1] = 31
    # print(generated.shape)
    # print(generated / 255)

    generated /= 255
    nearest = evaluation.to_nearest_label_color(generated)
    print(nearest[:, 0, 0])
    print(nearest[:, 0, 1])
    print(nearest[:, 1, 0])
    print(nearest[:, 1, 1])
    print(nearest[:, 2, 0])
    print(nearest[:, 2, 1])


def test_transient0():
    import data_handler
    import os

    # print(os.listdir('.'))
    # input()
    annotations, webcams = data_handler.read_annotations("data/transient/annotations/annotations.tsv")

    ind_spring, ind_winter = 2, 19
    ind_day, ind_night = 2, 3
    ind_warm, inc_cold = 11, 12

    # CREATE LIST OF ALL FEATURES
    # TRAIN TEST SPLIT
    all_uniques = []
    all_names = []

    # for ind in [ind_spring, ind_winter, ind_day, ind_night, ind_warm, inc_cold]:
    for ind in [ind_day, ind_night]:
        # 2 daylight, 3 night, 11 warm, 12 cold, 16 spring, 18 autumn, 19 winter
        imgs = data_handler.retrieve_imgs_with_attribute(annotations, ind)
        names = [img[0] for img in imgs]
        uniques = list(set([name.split('/')[0] for name in names]))

        all_names.append(names)
        all_uniques.append(uniques)

    # min_len = min([len(unique) for unique in uniques])
    # equal = [all_uniques[0][i] == all_uniques[1][i] for i in range(min_len)]

    # equal = set(all_uniques[0]) & set(all_uniques[1]) & set(all_uniques[2]) & \
     #       set(all_uniques[3]) & set(all_uniques[4]) & set(all_uniques[5])

    equals = list(set(all_uniques[0]) & set(all_uniques[1]))
    print('equal:', len(equals))

    for unique in equals:
        days = []
        nights = []

        print()
        input()

        # permute


def test_transient():
    import data_handler
    minorities = ['spring2autumn', 'summer2winter', 'dry2rain']  # then this
    type_2 = ['warm2cold', 'sunny2clouds']   # first this


    # pairs = data_handler.create_pairs_for_direction("data/transient/annotations/annotations.tsv", 'summer2winter', 'train')
    # print(pairs)
    # input()

    all = 0
    for direction in minorities:
        print('Doing for direction:', direction)
        train_pairs = data_handler.create_pairs_for_direction("data/transient/annotations/annotations.tsv", direction, split='train')
        val_pairs = data_handler.create_pairs_for_direction("data/transient/annotations/annotations.tsv", direction, split='val')
        test_pairs = data_handler.create_pairs_for_direction("data/transient/annotations/annotations.tsv", direction, split='test')

        all += len(train_pairs) + len(val_pairs) + len(test_pairs)
        print(f'all in all pairs: {len(train_pairs) + len(val_pairs) + len(test_pairs):,}')

    print('ALL:', all)


def test_datast_transient():
    params = helper.read_params('params.json')['transient']
    params['paths'] = params['local_paths']

    train_loader, _, _ = data_handler.init_transient_loaders(params)

    for i_batch, batch in enumerate(train_loader):
        im = torch.cat([batch['left'], batch['right']], dim=0)
        utils.save_image(im, f'tmp/0_transient/{i_batch}.png')

        if i_batch > 20:
            break


def eval_single_image():
    # syn_img = "/Midgard/home/sorkhei/glow2/samples/cityscapes/256x256/model=c_flow/img=segment/cond=real" \
    #           "/baseline + act_cond + w_cond + coupling_net/from_scratch/infer/step=83000/random_samples" \
    #           "/strasbourg_000001_061472/temp=0.7/sample 1.png"
    #
    # ref_img = "/Midgard/home/sorkhei/glow2/samples/cityscapes/256x256/model=c_flow/img=segment/cond=real" \
    #           "/baseline + act_cond + w_cond + coupling_net/from_scratch/infer/step=83000/random_samples" \
    #           "/strasbourg_000001_061472/temp=0.7/segmentation.png"

    # evaluation.evaluate_single_img(syn_img, ref_img)
    # evaluation.evaluate_single_img()
    pass


def main():
    # which_fn = 'initialize'
    # test_actnorm(which_fn)
    # test_read_cityscapes()

    # test_read_mnist()
    # test_save_mnist()
    # test_resume_train()
    # test_label_mnist()

    # test_resample()
    # test_city('train')
    # test_city('c_flow')

    # test_eval()
    # test_resize()

    # test_cn()
    # test_edge()

    # test_recreate_bmap()
    # test_pil()

    # test_cond_actnorm()
    # test_cond_net()
    # test_cond_coupling()

    # test_ssim_from_skimage()
    # eval_seg()
    # test_transient()
    # test_datast_transient()

    eval_seg()


if __name__ == '__main__':
    main()

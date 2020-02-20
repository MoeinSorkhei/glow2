import torch
import model
import data_handler
import train
import experiments
import helper

import matplotlib.pyplot as plt
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        glow = model.init_glow(params, cond_shapes)
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
            inp = glow.reverse(z, reconstruct=True, cond=cond)
            print('backward first done')

            inp_rand = glow.reverse(z_rand, reconstruct=False, cond=cond)
            print('backward second done')


def main():
    # which_fn = 'initialize'
    # test_actnorm(which_fn)
    # test_read_cityscapes()

    # test_read_mnist()
    # test_save_mnist()
    # test_resume_train()
    # test_label_mnist()

    # test_resample()
    test_city('train')


if __name__ == '__main__':
    main()

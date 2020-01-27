from comet_ml import Experiment

from model import Glow
import helper_prev
from helper import init_comet
import data_handler

from tqdm import tqdm
from math import log
import argparse
import json
import os

import torch
from torchvision import utils
from torch import nn, optim


# determine the device globally for every function to use it if needed
global_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_params_and_args():
    parser = argparse.ArgumentParser(description='Glow trainer')
    parser.add_argument('--batch', default=2, type=int, help='batch size')  # 256 => 2, 128 => 8, 64 => 16
    parser.add_argument('--dataset', type=str, help='the name of the dataset')

    # parser.add_argument('--iter', default=200000, type=int, help='maximum iterations')
    # parser.add_argument(
    #    '--n_flow', default=32, type=int, help='number of flows in each block'
    # )
    # parser.add_argument('--n_block', default=4, type=int, help='number of blocks')
    # parser.add_argument(
    #     '--no_lu',
    #     action='store_true',
    #     help='use plain convolution instead of LU decomposed version',
    # )
    # parser.add_argument(
    #     '--affine', action='store_true', help='use affine coupling instead of additive'
    # )
    # parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
    # parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--img_size', default=256, type=int, help='image size')
    # parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
    # parser.add_argument('--n_sample', default=30, type=int, help='number of samples')
    # parser.add_argument('path', metavar='PATH', type=str, help='Path to image directory')

    parser.add_argument('--use_comet', action='store_true')
    parser.add_argument('--resume_train', action='store_true')
    arguments = parser.parse_args()

    # reading params from the json file
    with open('params.json', 'r') as f:
        parameters = json.load(f)

    return arguments, parameters


def calc_loss(log_p, logdet, image_size, n_bins):  # how does it work
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def resume_training(glow, opt, optim_step, arguments, parameters, input_channels, comet_tracker):
    model_path = f'checkpoints/model_{str(optim_step).zfill(6)}.pt'
    optim_path = f'checkpoints/optim_{str(optim_step).zfill(6)}.pt'

    glow.load_state_dict(torch.load(model_path))
    glow = nn.DataParallel(glow)
    # print('model load')
    # input()

    opt.load_state_dict(torch.load(optim_path))

    # print('optim load')
    # input()

    # need to also save the z variables if we want to resume with exactly the same samples +
    # optim_step, or epoch + last train loss
    glow.train()  # set to train mode
    train(arguments, parameters, glow, opt, input_channels, comet_tracker, resume=True, last_optim_step=optim_step)


def train(args, params, model, optimizer, in_channels, comet_tracker=None, resume=False, last_optim_step=0):
    # dataset_name = 'cityscapes_segmentation'
    # dataset_name = 'cityscapes_leftImg8bit'
    # loader_params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': 4}
    loader_params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': 4}
    data_loader = data_handler.init_data_loader(data_folder=params['datasets'][args.dataset],
                                                dataset_name=args.dataset,
                                                image_size=args.img_size,
                                                remove_alpha=True,  # removing the alpha channel not used for generation
                                                loader_params=loader_params)
    n_bins = 2. ** params['n_bits']
    z_sample = []
    z_shapes = helper_prev.calc_z_shapes(in_channels, args.img_size, params['n_flow'], params['n_block'])

    for z in z_shapes:  # temperature squeezes the Gaussian which is sampled from
        z_new = torch.randn(params['n_samples'], *z) * params['temperature']
        z_sample.append(z_new.to(device))

    optim_step = last_optim_step if resume else 0
    max_optim_steps = params['iter']

    if resume:
        print(f'In [train]: resuming training from optim_step={optim_step}')

    print(f'In [train]: sarting training with a data loader of size: {len(data_loader)}')
    while optim_step < max_optim_steps:
        for i_batch, batch in enumerate(data_loader):
            batch = batch.to(device)

            # print(batch.shape)
            # print(batch)
            # print(model(batch))
            # print('before else')
            # input()

            # I do not know this
            if optim_step == 0:
                with torch.no_grad():  # why
                    log_p, logdet, _ = model.module(batch + torch.rand_like(batch) / n_bins)
                    optim_step += 1
                    continue
            else:
                log_p, logdet, _ = model(batch + torch.rand_like(batch) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = params['lr']
            optimizer.param_groups[0]['lr'] = warmup_lr
            optimizer.step()

            print(f'Step: {optim_step} => loss: {loss.item():.3f}, log_p: {log_p.item():.3f}')

            if args.use_comet:
                comet_tracker.track_metric('loss', round(loss.item(), 3), optim_step)

            if optim_step % 100 == 0:
                if not os.path.exists('samples'):
                    os.mkdir('samples')
                    print('In [train]: created path "samples"...')

                with torch.no_grad():
                    # print(z_shapes)
                    # print(model_single.reverse(z_sample).cpu().shape)

                    # why .CPU?
                    sampled_images = model_single.reverse(z_sample).cpu().data
                    utils.save_image(sampled_images, f'samples/{str(optim_step).zfill(6)}.png', nrow=10)

                    '''utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f'sample/{str(optim_step).zfill(6)}.png',
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )'''
                print("\nSample saved at iteration", optim_step, '\n')

            if optim_step % 1000 == 0:
                if not os.path.exists('checkpoints'):
                    os.mkdir('checkpoints')
                    print('In [train]: created path "checkpoints"...')

                torch.save(model.state_dict(), f'checkpoints/model_{str(optim_step).zfill(6)}.pt')
                torch.save(optimizer.state_dict(), f'checkpoints/optim_{str(optim_step).zfill(6)}.pt')
                print("Checkpoint saved at iteration", optim_step, '\n')
            optim_step += 1


def train_prev(args, model, optimizer):
    dataset = iter(helper_prev.sample_data(args.path, args.batch, args.img_size))
    n_bins = 2. ** args.n_bits

    z_sample = []
    z_shapes = helper_prev.calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset)
            image = image.to(device)

            # I do not know this
            if i == 0:
                with torch.no_grad():  # why
                    log_p, logdet, _ = model.module(image + torch.rand_like(image) / n_bins)
                    continue
            else:
                log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]['lr'] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f'Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}'
            )

            if i % 100 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f'sample/{str(i + 1).zfill(6)}.png',
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )
                print("\nSample saved at iteration", i, '\n')

            if i % 10000 == 0:
                torch.save(
                    model.state_dict(), f'checkpoint/model_{str(i).zfill(6)}.pt'
                )
                torch.save(
                    optimizer.state_dict(), f'checkpoint/optim_{str(i).zfill(6)}.pt'
                )
                print("Checkpoint saved at iteration", i, '\n')


if __name__ == '__main__':
    args, params = read_params_and_args()
    print('In [main]: arguments:', args)

    # initializing the model and the optimizer
    # RGB images, if image is PNG, the alpha channel will be removed
    in_channels = 3
    model_single = Glow(
        in_channels, params['n_flow'], params['n_block'], do_affine=params['affine'], conv_lu=params['lu']
    )
    model = nn.DataParallel(model_single)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    tracker = None
    if args.use_comet:
        # experiment = Experiment(api_key="QLZmIFugp5kqZjA4XE2yNS0iZ", project_name="glow", workspace="moeinsorkhei")
        run_params = {'data_folder': params['datasets'][args.dataset]}
        tracker = init_comet(run_params)
        print("Comet experiment initialized...")

    # resume training
    if args.resume_train:
        optim_step = 6000
        resume_training(model, optimizer, optim_step, args, params, in_channels, tracker)

    # train from scratch
    else:
        train(args, params, model, optimizer, in_channels, tracker)
        # train(args, params, model, optimizer, in_channels, tracker)
    # else:
      #  train(args, params, model, optimizer, in_channels)
    # train_prev(args, model, optimizer)

"""
Run in GCP:
python train.py cityscapes/gtFine_trainvaltest/gtFine/train/aachen --affine --use_comet
python train.py cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/aachen --affine


Note: 
If the dataset is celeba_limted: path should be 'data/celeba_limited'
If it is cityscapes, then it should be the path starting from the data directory: 
    - for instance: 'cityscapes/gtFine_trainvaltest/gtFine/train/aachen'


For running in terminal on my Mac with path: ../../Desktop/KTH/RA/Datasets/celeba_limited/class1
/Users/user/.conda/envs/ADL/bin/python /Users/user/PycharmProjects/glow2/train.py data/celeba_limited --affine
"""
# path: ../../Desktop/KTH/RA/Datasets/celeba_limited/class1
# /Users/user/.conda/envs/ADL/bin/python /Users/user/PycharmProjects/glow2/train.py

import torch
from . import model, data_handler, train
from . import experiments
import json


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
    dataset = data_handler.CityDataset(data_folder=seg_folder, dataset_name='cityscapes_segmentation')
    print(len(dataset))


def test_read_mnist():
    experiments.visualize_mnist()


def test_save_mnist():
    experiments.save_mnist_rgb('data/mnist', 'data/mnist/images')


def test_resume_train():
    # reading params from the json file
    with open('params.json', 'r') as f:
        parameters = json.load(f)['mnist']  # parameters related to the wanted dataset

    optim_step = 9000
    device = torch.device('cpu')
    train.resume_train(optim_step, parameters, device)


def main():
    # which_fn = 'initialize'
    # test_actnorm(which_fn)
    # test_read_cityscapes()
    # test_read_mnist()
    # test_save_mnist()
    test_resume_train()


if __name__ == '__main__':
    main()


# --dataset mnist --batch 128 --img_size 24
# --dataset mnist --batch 128 --img_size 24 --resume_train --last_optim_step 9000

# --dataset cityscapes_segmentation
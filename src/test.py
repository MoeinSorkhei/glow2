import torch
import model
import data_handler
import train
import experiments
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


def main():
    # which_fn = 'initialize'
    # test_actnorm(which_fn)
    # test_read_cityscapes()

    # test_read_mnist()
    # test_save_mnist()
    # test_resume_train()
    # test_label_mnist()

    l = ['2']
    test_list(l)
    print('l:', l)


if __name__ == '__main__':
    main()

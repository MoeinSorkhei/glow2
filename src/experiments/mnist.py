from .. import data_handler
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_mnist(mnist_folder):
    # mnist_folder = 'data/mnist'
    imgs, labels = data_handler.read_mnist(mnist_folder)

    index = 100  # index of the image to be visualized
    img = imgs[index].reshape(28, 28)
    label = labels[index]

    # plotting
    plt.title(f'Label is: {label}')
    plt.imshow(img, cmap='gray')
    plt.show()


def save_mnist_rgb(mnist_folder, save_folder):
    imgs, labels = data_handler.read_mnist(mnist_folder)

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # save 1000 images
    for index in range(1000):
        img = imgs[index].reshape(28, 28) / 255

        rgb = np.zeros((28, 28, 3))
        rgb[:, :, 0] = img
        rgb[:, :, 1] = img
        rgb[:, :, 2] = img
        plt.imsave(save_folder + f'/{index}.jpg', rgb)

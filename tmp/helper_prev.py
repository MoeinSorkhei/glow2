from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def sample_data(path, batch_size, image_size):
    """
    Todo:
        - the 'yield' part of this function is written a bit tricky and should be enhanced.
    Note: I se num_workers to 0
    Samples images from the image path, and performs som transformations on the image.
    :param path:
    :param batch_size:
    :param image_size:
    :return:
    """
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=0
            )
            loader = iter(loader)
            yield next(loader)

from torch.utils import data
from torchvision import transforms

from PIL import Image
import os


class MapsDataset(data.Dataset):
    def __init__(self, data_folder, split, img_size, fixed_conds=None):
        self.data_folder = data_folder
        self.split = split
        self.img_size = img_size
        self.fixed_conds = fixed_conds
        self.imgs = read_imgs(data_folder, split, fixed_conds)
        self.transforms = init_transformers(img_size)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        image = self.transforms(Image.open(self.imgs[item]))
        h, w = self.img_size[0], self.img_size[1]
        photo = image[:, :, :w]
        the_map = image[:, :, w:]

        return {
            'photo': photo,
            'the_map': the_map
        }


def init_transformers(img_size):
    # the images are sticked together, ie, 512x512 -> 512x1024 (see some images to understand this)
    trans = transforms.Compose([transforms.Resize((img_size[0], img_size[1] * 2)), transforms.ToTensor()])
    return trans


def read_imgs(data_folder, split, fixed_conds):
    data_split_path = os.path.join(data_folder, split)  # path for the data split

    if fixed_conds is not None:
        files = [os.path.join(data_split_path, fixed_conds[i]) for i in range(len(fixed_conds))]

    else:  # read all the images in the path
        files = [os.path.join(data_split_path, file) for file in os.listdir(data_split_path)]

    # print(f'In [read_imgs]: read {len(files)} files in path: "{data_split_path}": done')
    return files


def init_loaders(loader_params, dataset_params):
    loaders = []
    for split in ['train', 'val']:
        dataset_params['split'] = split
        loader_params['shuffle'] = True if split == 'train' else False

        dataset = MapsDataset(**dataset_params)
        data_loader = data.DataLoader(dataset, **loader_params)
        loaders.append(data_loader)
    return loaders


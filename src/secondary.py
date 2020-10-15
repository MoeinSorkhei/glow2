import helper
import os
import numpy as np
from PIL import Image
import experiments


def custom_round(num):
    return round(num * 100, 2)


def extract_psp(temp):
    base_path = '/Users/user/PycharmProjects/glow2/samples/cityscapes/256x256/improved_1_regular_lu_lambda_0.0001/label2photo/infer/step=136000/eval'
    mIoU_list, mAcc_list, allAcc_list = [], [], []

    for sampling_round in [1, 2, 3]:
        path = os.path.join(base_path, f'temp={temp}', f'val_imgs_{sampling_round}_eval', 'psp_score.txt')
        # print('Reading:', path)
        # input()

        line = helper.read_file_to_list(path)[1][:-1]
        splits = line.split('/')
        iou, m_acc, a_acc = float(splits[0]), float(splits[1]), float(splits[2])
        # print('iou, m_acc, a_acc:', iou, m_acc, a_acc)
        # input()

        mIoU_list.append(iou)
        mAcc_list.append(m_acc)
        allAcc_list.append(a_acc)

    print('Results for temp', temp)
    # print(f'{np.mean(mIoU_list)} +/- {np.std(mIoU_list)}, {np.mean(mAcc_list)} +/- {np.std(mAcc_list)}, {np.mean(allAcc_list)} +/- {np.std(allAcc_list)}')
    print(f'{custom_round(np.mean(allAcc_list))} +/- {custom_round(np.std(allAcc_list))}, '
          f'{custom_round(np.mean(mAcc_list))} +/- {custom_round(np.std(mAcc_list))}, '
          f'{custom_round(np.mean(mIoU_list))} +/- {custom_round(np.std(mIoU_list))}')
    # print(f'{np.std(mIoU_list)}, {np.std(mAcc_list)}, {np.std(allAcc_list)}')


def separate_images():
    # for im_mode in ['segment', 'real']:
    dest1 = f'/Midgard/Data/moein/cityscapes/downsampled/all_segments'
    dest2 = f'/Midgard/Data/moein/cityscapes/downsampled/all_reals'
    helper.make_dir_if_not_exists(dest1)
    helper.make_dir_if_not_exists(dest2)

    # for mode in ['train', 'test']:
    source = f'/Midgard/Data/moein/cityscapes/downsampled/all_combined'
    files = helper.files_with_suffix(source, '.png')
    print(f'Read files of len:', len(files))

    for i_file, file in enumerate(files):
        image = np.array(Image.open(file))
        seg, real = image[:, :256, :], image[:, 256:, :]
        name = os.path.split(file)[-1].replace('_combined.png', '.png')

        save_path1 = os.path.join(dest1, name)
        save_path2 = os.path.join(dest2, name)

        Image.fromarray(seg).save(save_path1)
        Image.fromarray(real).save(save_path2)
        print('Done for', i_file)


def down_sample_city():
    for mode in ['train', 'test']:
        # folder = f'../data/cityscapes_orig_size/{mode}'
        # dest = f'../data/cityscapes_downsampled'
        folder = f'/Midgard/Data/moein/cityscapes/combined/{mode}'
        dest = f'/Midgard/Data/moein/cityscapes/downsampled'

        print('Reading:', folder)
        files = helper.files_with_suffix(folder, '.png')
        print(f'Read files of len:', len(files))

        for i_file, file in enumerate(files):
            # image = np.array(Image.open(file))
            # print(image.shape)
            helper.make_dir_if_not_exists(dest)
            save_path = os.path.join(dest, os.path.split(file)[-1])
            Image.open(file).resize((512, 256)).save(save_path)
            print('Done for', i_file)


def transfer(args, params):
    content_basepath = '../data/cityscapes_complete_downsampled/all_reals'
    cond_basepath = '../data/cityscapes_complete_downsampled/all_segments'

    # pure_content = 'jena_000011_000019'
    # pure_new_cond = 'jena_000066_000019'

    # pure_content = 'aachen_000028_000019'
    # pure_new_cond = 'jena_000011_000019'

    # pure_content = 'jena_000011_000019'
    # pure_new_cond = 'aachen_000010_000019'

    pure_content = 'aachen_000034_000019'
    pure_new_cond = 'bochum_000000_016260'

    content = f'{content_basepath}/{pure_content}.png'  # content image
    condition = f'{cond_basepath}/{pure_content}.png'   # corresponding condition needed to extract z
    new_cond = f'{cond_basepath}/{pure_new_cond}.png'   # new condition

    save_basepath = '../samples/content_transfer_local'
    helper.make_dir_if_not_exists(save_basepath)
    file_path = f'{save_basepath}/content={pure_content}_condition={pure_new_cond}.png'
    experiments.transfer_content(args, params, content, condition, new_cond, file_path)


def main(args, params):
    # for t in [0.4, 0.3, 0.2, 0.1, 0.0]:
    #     extract_psp(temp=t)
    # extract_psp(temp=0.8)
    # down_sample_city()
    # separate_images()
    transfer(args, params)


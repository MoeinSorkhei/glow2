from globals import device


def extract_batches(batch, args):
    """
    This function depends onf the dataset and direction.
    :param batch:
    :param args:
    :return:
    """
    if args.dataset == 'cityscapes':
        img_batch = batch['real'].to(device)
        segment_batch = batch['segment'].to(device)
        boundary_batch = batch['boundary'].to(device) if args.use_bmaps else None

    elif args.dataset == 'maps':
        img_batch = batch['photo'].to(device)
        segment_batch = batch['the_map'].to(device)
        boundary_batch = None

    else:
        raise NotImplementedError
    return img_batch, segment_batch, boundary_batch

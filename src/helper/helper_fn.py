import torch


def save_checkpoint(path_to_save, optim_step, model, optimizer, loss):
    name = path_to_save + f'/optim_step={optim_step}.pt'
    checkpoint = {'loss': loss,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, name)


def load_checkpoint(path_to_load, optim_step, model, optimizer, device, resume_train=True):
    # path_to_load = translate_address(path_to_load, 'helper')
    name = path_to_load + f'/optim_step={optim_step}.pt'
    checkpoint = torch.load(name, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']

    print('In [load_checkpoint]: load state dict: done')

    # putting the model in the correct mode
    model.train() if resume_train else model.eval()  # model or model_single?
    # model_single.train() if resume_train else model_single.eval()

    # return model_single, model, optimizer, loss
    return model, optimizer, loss


# def load_model_and_optimizer(model, model_single, optimizer, model_path, optim_path, device, resume_train=True):
#     """
#     I believe this function is not longer gonna be used.
#     :param model:
#     :param model_single:
#     :param optimizer:
#     :param model_path:
#     :param optim_path:
#     :param device:
#     :param resume_train:
#     :return:
#     """
#     # model_single.load_state_dict(torch.load(model_path, map_location=device))
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     optimizer.load_state_dict(torch.load(optim_path, map_location=device))
#     print('In [load_model_and_optimizer]: load state dict done')
#
#     # putting the model in the correct mode
#     model.train() if resume_train else model.eval()  # model or model_single?
#     model_single.train() if resume_train else model_single.eval()
#
#     return model_single, model, optimizer


def translate_address(path, package):
    """
    This function changes a path which is from the project directory to a path readable by a specific package in the
    folder. For instance, the datasets paths in params.json are from the project directory. Using this function, every
    function in the data_loader package can use that address by converting it to a relative address using this function.
    The benefit is that no function needs to translate the address itself directly, and all of them could use this
    function to do so.
    :param path:
    :param package:
    :return:
    """
    if package == 'data_handler' or package == 'helper':
        return '../' + path
    else:
        raise NotImplementedError('NOT IMPLEMENTED...')

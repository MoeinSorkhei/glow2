from . import city, maps, mnist, transient
from globals import maps_fixed_conds


def retrieve_rev_cond(args, params, run_mode='train'):
    if args.dataset == 'mnist':
        raise NotImplementedError
        # model = init_glow(params)
        # reverse_cond = prepare_city_reverse_cond(args, params, run_mode)  # NOTE: needs to be refactored
        # reverse_cond = None  # NOTE: needs to be refactored

    elif args.dataset == 'cityscapes':
        reverse_cond = None if args.exp else city.prepare_city_reverse_cond(args, params, run_mode)
        # mode = None  # no mode if 'photo2label'
        # if args.direction == 'label2photo':
        #     mode = 'segment_boundary' if args.use_bmaps else 'segment'

    elif args.dataset == 'maps':
        reverse_cond = maps.create_rev_cond(args, params, fixed_conds=maps_fixed_conds, also_save=True)
        mode = None
    else:
        raise NotImplementedError('Dataset not implemented')
    return reverse_cond

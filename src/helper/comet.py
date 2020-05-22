from comet_ml import Experiment, ExistingExperiment


class CometTracker:
    def __init__(self, comet_params, run_params=None, prev_exp_id=None):
        if prev_exp_id:  # previous experiment
            api_key = comet_params['api_key']
            del comet_params['api_key']  # removing this because the rest of the items need to be passed
            self.experiment = ExistingExperiment(api_key=api_key,
                                                 previous_experiment=prev_exp_id,
                                                 **comet_params)
            print(f'In CometTracker: ExistingExperiment initialized with id: {prev_exp_id}')

        else:  # new experiment
            self.experiment = Experiment(**comet_params)
            self.experiment.log_parameters(run_params)

    def track_metric(self, metric, value, step):
        self.experiment.log_metric(metric, value, step)

    def add_tags(self, tags):
        self.experiment.add_tags(tags)
        print(f'In [add_tags]: Added these tags to the new experiment: {tags}')


def init_comet(args, params=None):
    """
    This function uses the comet_ml package to track the experiment.
    :return: the comet experiment
    """
    # params for Moein
    comet_params = {
        'api_key': "QLZmIFugp5kqZjA4XE2yNS0iZ",
        'project_name': "glow",
        'workspace': "moeinsorkhei"
    }

    # create the comet tracker
    tracker = CometTracker(comet_params, run_params=params, prev_exp_id=args.prev_exp_id)

    # create tags for the new experiment
    if not args.prev_exp_id:
        tags = create_tags(args, params)
        tracker.add_tags(tags)

    return tracker


def create_tags(args, params):
    tags = [f'{params["img_size"][0]}x{params["img_size"][1]}']  # image size
    if args.w_conditional:  # w_conditional
        tags.append('w_cond')

    if args.act_conditional:  # act_conditional
        tags.append('act_cond')

    if args.cond_mode == 'segment_boundary':  # boundary
        tags.append('boundary')

    if args.cond_mode == 'segment':  # segment
        tags.append('segment')

    if args.coupling_cond_net:
        tags.append('coupling_net')

    if args.left_pretrained:  # left_pretrained
        tags.append('left_pretrained')
    else:  # from scratch
        tags.append('scratch')
    return tags

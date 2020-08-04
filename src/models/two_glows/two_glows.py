from ..glow import *
import helper


class TwoGlows(nn.Module):
    def __init__(self, args, params):
        super().__init__()
        left_configs, right_configs = init_configs(args)
        split_type = right_configs['split_type']
        input_shapes = calc_inp_shapes(params['channels'], params['img_size'], params['n_block'], split_type)
        cond_shapes = calc_cond_shapes(params['channels'], params['img_size'], params['n_block'], split_type)  # shape (C, H, W)
        # print_all_shapes(input_shapes, cond_shapes, params, split_type)

        self.left_glow = init_glow(n_blocks=params['n_block'],
                                   n_flows=params['n_flow'],
                                   input_shapes=input_shapes,
                                   cond_shapes=None,
                                   configs=left_configs)

        self.right_glow = init_glow(n_blocks=params['n_block'],
                                    n_flows=params['n_flow'],
                                    input_shapes=input_shapes,
                                    cond_shapes=cond_shapes,
                                    configs=right_configs)

    def forward(self, x_a, x_b, b_map=None):  # x_a: segmentation
        #  perform left glow forward
        left_glow_out = self.left_glow(x_a)

        # extract left outputs
        log_p_sum_left, log_det_left = left_glow_out['log_p_sum'], left_glow_out['log_det']
        z_outs_left, flows_outs_left = left_glow_out['z_outs'], left_glow_out['all_flows_outs']

        # perform right glow forward
        conditions = prep_conds(left_glow_out, direction='forward')
        right_glow_out = self.right_glow(x_b, conditions)

        # extract right outputs
        log_p_sum_right, log_det_right = right_glow_out['log_p_sum'], right_glow_out['log_det']
        z_outs_right, flows_outs_right = right_glow_out['z_outs'], right_glow_out['all_flows_outs']

        # gather left outputs together
        left_glow_outs = {'log_p': log_p_sum_left, 'log_det': log_det_left,
                          'z_outs': z_outs_left, 'flows_outs': flows_outs_left}

        #  gather right outputs together
        right_glow_outs = {'log_p': log_p_sum_right, 'log_det': log_det_right,
                           'z_outs': z_outs_right, 'flows_outs': flows_outs_right}

        return left_glow_outs, right_glow_outs

    def reverse(self, x_a=None, z_b_samples=None):
        left_glow_out = self.left_glow(x_a)
        conditions = prep_conds(left_glow_out, direction='reverse')
        x_b_syn = self.right_glow.reverse(z_b_samples, conditions=conditions)  # sample x_b conditioned on x_a
        return x_b_syn

    def new_condition(self, x_a, z_b_samples):
        left_glow_out = self.left_glow(x_a)
        conditions = prep_conds(left_glow_out, direction='reverse')
        x_b_rec = self.right_glow.reverse(z_b_samples, reconstruct=True, conditions=conditions)
        return x_b_rec

    def reconstruct_all(self, x_a, x_b):
        left_glow_out = self.left_glow(x_a)
        z_outs_left = left_glow_out['z_outs']
        print('left forward done')

        conditions = prep_conds(left_glow_out, direction='forward')  # preparing for right glow forward
        right_glow_out = self.right_glow(x_b, conditions)
        z_outs_right = right_glow_out['z_outs']
        print('right forward done')

        # reverse operations
        x_a_rec = self.left_glow.reverse(z_outs_left, reconstruct=True)
        print('left reverse done')

        conditions = prep_conds(left_glow_out, direction='reverse')  # prepare for right glow reverse
        x_b_rec = self.right_glow.reverse(z_outs_right, reconstruct=True, conditions=conditions)
        print('right reverse done')
        return x_a_rec, x_b_rec


def print_all_shapes(input_shapes, cond_shapes, params, split_type):
    helper.print_and_wait(f'input_shapes: {input_shapes}')
    helper.print_and_wait(f'cond_shapes: {cond_shapes}')
    z_shapes = calc_z_shapes(params['channels'], params['img_size'], params['n_block'], split_type)
    helper.print_and_wait(f'z_shapes: {z_shapes}')


def init_configs(args):
    left_configs = {'all_conditional': False, 'split_type': 'regular'}  # default
    right_configs = {'all_conditional': True, 'split_type': 'regular'}  # default

    if 'improved_1' in args.model:
        left_configs['split_type'] = 'special'
        right_configs['split_type'] = 'special'
        left_configs['split_sections'] = [3, 9]
        right_configs['split_sections'] = [3, 9]

    return left_configs, right_configs


def prep_conds(left_glow_out, direction):
    left_glow_w_outs = left_glow_out['all_w_outs']
    left_glow_act_outs = left_glow_out['all_act_outs']
    left_coupling_outs = left_glow_out['all_flows_outs']
    conditions = make_cond_dict(left_glow_act_outs, left_glow_w_outs, left_coupling_outs)

    if direction == 'reverse':  # reverse lists
        conditions['act_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['act_cond']))]  # reverse 2d list
        conditions['w_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['w_cond']))]
        conditions['coupling_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['coupling_cond']))]
    return conditions


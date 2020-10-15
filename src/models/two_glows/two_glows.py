import torch

from ..glow import *
import helper


class TwoGlows(nn.Module):
    def __init__(self, params, left_configs, right_configs):
        super().__init__()
        self.left_configs, self.right_configs = left_configs, right_configs

        self.split_type = right_configs['split_type']  # this attribute will also be used in take sample
        condition = right_configs['condition']
        input_shapes = calc_inp_shapes(params['channels'],
                                       params['img_size'],
                                       params['n_block'],
                                       self.split_type)

        cond_shapes = calc_cond_shapes(params['channels'],
                                       params['img_size'],
                                       params['n_block'],
                                       self.split_type,
                                       condition)  # shape (C, H, W)

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

    def prep_conds(self, left_glow_out, b_map, direction):
        act_cond = left_glow_out['all_act_outs']
        w_cond = left_glow_out['all_w_outs']  # left_glow_out in the forward direction
        coupling_cond = left_glow_out['all_flows_outs']

        # important: prep_conds will change the values of left_glow_out, so left_glow_out is not valid after this function
        cond_config = self.right_configs['condition']
        if 'b_maps' in cond_config:
            for block_idx in range(len(act_cond)):
                for flow_idx in range(len(act_cond[block_idx])):
                    cond_h, cond_w = act_cond[block_idx][flow_idx].shape[2:]
                    do_ceil = 'ceil' in cond_config

                    # helper.print_and_wait(f'b_map size: {b_map.shape}')
                    # b_map_cond = helper.resize_tensor(b_map.squeeze(dim=0), (cond_w, cond_h), do_ceil).unsqueeze(dim=0)  # resize
                    b_map_cond = helper.resize_tensors(b_map, (cond_w, cond_h), do_ceil)  # resize

                    # concat channel wise
                    act_cond[block_idx][flow_idx] = torch.cat(tensors=[act_cond[block_idx][flow_idx], b_map_cond], dim=1)
                    w_cond[block_idx][flow_idx] = torch.cat(tensors=[w_cond[block_idx][flow_idx], b_map_cond], dim=1)
                    coupling_cond[block_idx][flow_idx] = torch.cat(tensors=[coupling_cond[block_idx][flow_idx], b_map_cond], dim=1)

        # make conds a dictionary
        conditions = make_cond_dict(act_cond, w_cond, coupling_cond)

        # reverse lists for reverse operation
        if direction == 'reverse':
            conditions['act_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['act_cond']))]  # reverse 2d list
            conditions['w_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['w_cond']))]
            conditions['coupling_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['coupling_cond']))]
        return conditions

    def forward(self, x_a, x_b, extra_cond=None):  # x_a: segmentation
        #  perform left glow forward
        left_glow_out = self.left_glow(x_a)

        # perform right glow forward
        conditions = self.prep_conds(left_glow_out, extra_cond, direction='forward')
        right_glow_out = self.right_glow(x_b, conditions)

        # extract left outputs
        log_p_sum_left, log_det_left = left_glow_out['log_p_sum'], left_glow_out['log_det']
        z_outs_left, flows_outs_left = left_glow_out['z_outs'], left_glow_out['all_flows_outs']

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

    def reverse(self, x_a=None, z_b_samples=None, extra_cond=None, reconstruct=False):
        left_glow_out = self.left_glow(x_a)  # left glow forward always needed before preparing conditions
        conditions = self.prep_conds(left_glow_out, extra_cond, direction='reverse')
        x_b_syn = self.right_glow.reverse(z_b_samples, reconstruct=reconstruct, conditions=conditions)  # sample x_b conditioned on x_a
        return x_b_syn

    def new_condition(self, x_a, z_b_samples):
        left_glow_out = self.left_glow(x_a)
        conditions = self.prep_conds(left_glow_out, b_map=None, direction='reverse')  # should be tested
        x_b_rec = self.right_glow.reverse(z_b_samples, reconstruct=True, conditions=conditions)
        return x_b_rec

    def reconstruct_all(self, x_a, x_b, b_map=None):
        left_glow_out = self.left_glow(x_a)
        print('left forward done')

        z_outs_left = left_glow_out['z_outs']
        conditions = self.prep_conds(left_glow_out, b_map, direction='forward')  # preparing for right glow forward
        right_glow_out = self.right_glow(x_b, conditions)
        z_outs_right = right_glow_out['z_outs']
        print('right forward done')

        # reverse operations
        x_a_rec = self.left_glow.reverse(z_outs_left, reconstruct=True)
        print('left reverse done')
        
        # need to do forward again since left_glow_out has been changed after preparing condition
        left_glow_out = self.left_glow(x_a)
        conditions = self.prep_conds(left_glow_out, b_map, direction='reverse')  # prepare for right glow reverse
        x_b_rec = self.right_glow.reverse(z_outs_right, reconstruct=True, conditions=conditions)
        print('right reverse done')
        return x_a_rec, x_b_rec


def print_all_shapes(input_shapes, cond_shapes, params, split_type):  # for debugging
    z_shapes = calc_z_shapes(params['channels'], params['img_size'], params['n_block'], split_type)
    # helper.print_and_wait(f'z_shapes: {z_shapes}')
    # helper.print_and_wait(f'input_shapes: {input_shapes}')
    # helper.print_and_wait(f'cond_shapes: {cond_shapes}')
    print(f'z_shapes: {z_shapes}')
    print(f'input_shapes: {input_shapes}')
    print(f'cond_shapes: {cond_shapes}')

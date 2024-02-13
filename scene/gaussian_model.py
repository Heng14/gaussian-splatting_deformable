#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

import scene.rigid_body as rigid

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm

# by heng

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims, i=0):
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim



def default_init(tensor):
    return nn.init.xavier_uniform_(tensor)

def rotation_init(tensor):
    return nn.init.uniform_(tensor, -1e-4, 1e-4)

def pivot_init(tensor):
    return nn.init.uniform_(tensor, -1e-4, 1e-4)

def translation_init(tensor):
    return nn.init.uniform_(tensor, -1e-4, 1e-4)



class DirectTemporalNeRF_se3(nn.Module):

    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True):
        super(DirectTemporalNeRF_se3, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        self.embed_fn = embed_fn
        self.zero_canonical = zero_canonical
        self._time, self._w, self._v = self.create_time_net()
        
        branches = {
            'w':
                self._w,     
            'v':
                self._v,
        }
        self.branches = branches


    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3), nn.Linear(self.W, 3)



    def query_time(self, new_pts, t, net, net_w, net_v):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return net_w(h), net_v(h)


    def forward(self, x, ts, iteration):

        input_pts = x
        assert len(torch.unique(ts[:, :1])) == 1, "Only accepts all points from same time"
        cur_time = ts[0, 0]

        w, v = self.query_time(input_pts, ts, self._time, self._w, self._v)

        theta = torch.norm(w, dim=-1)
        w = w / theta[..., None]
        v = v / theta[..., None]
        screw_axis = torch.cat([w, v], dim=-1)
        transform = rigid.exp_se3(screw_axis, theta)


        # if cur_time == 0. : #or iteration < 3000: # by heng
        if iteration < 3000: # by heng
        # if False:
            transform = torch.zeros_like(input_pts[:, :4])

        return transform


# ## ori good
# class DirectTemporalNeRF(nn.Module):
#     def __init__(self, D=3, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
#                  use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True):
#         super(DirectTemporalNeRF, self).__init__()


#         self.embed_fn, self.input_ch = get_embedder(10, 3, 0) # 10
#         self.embedtime_fn, self.input_ch_time = get_embedder(10, 1, 0)

#         self.D = D
#         self.W = W
#         # self.input_ch = input_ch   # by heng
#         self.input_ch_views = input_ch_views
#         # self.input_ch_time = input_ch_time # by heng
#         self.skips = skips
#         self.use_viewdirs = use_viewdirs
#         self.memory = memory
#         # self.embed_fn = embed_fn # by heng
#         self.zero_canonical = zero_canonical
#         self._time, self._time_out = self.create_time_net()

#     def create_time_net(self):
#         layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
#         for i in range(self.D - 1):
#             if i in self.memory:
#                 raise NotImplementedError
#             else:
#                 layer = nn.Linear

#             in_channels = self.W
#             if i in self.skips:
#                 in_channels += self.input_ch

#             layers += [layer(in_channels, self.W)]
#         return nn.ModuleList(layers), nn.Linear(self.W, 3)

#     def query_time(self, new_pts, t, net, net_final):
#         h = torch.cat([new_pts, t], dim=-1)
#         for i, l in enumerate(net):
#             h = net[i](h)
#             h = F.relu(h)
#             if i in self.skips:
#                 h = torch.cat([new_pts, h], -1)

#         return net_final(h)

#     def forward(self, x, ts, iteration):
#         input_pts = x

#         # print ('iteration: ', iteration) # by heng

#         input_pts = self.embed_fn(input_pts) # by heng
#         ts = self.embedtime_fn(ts) # by heng

#         assert len(torch.unique(ts[:, :1])) == 1, "Only accepts all points from same time"
#         cur_time = ts[0, 0]
#         # if cur_time == 0. or iteration < 3000: # by heng
#         if iteration < 3000: # by heng
#         # if False:
#             dx = torch.zeros_like(input_pts[:, :3])
#         else:
#             dx = self.query_time(input_pts, ts, self._time, self._time_out)
#         return dx


class DirectTemporalNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True):
        super(DirectTemporalNeRF, self).__init__()


        self.embed_fn, self.input_ch = get_embedder(10, input_ch, 0) # 10
        self.embedtime_fn, self.input_ch_time = get_embedder(10, 1, 0)

        self.D = D
        self.W = W
        # self.input_ch = input_ch   # by heng
        self.input_ch_views = input_ch_views
        # self.input_ch_time = input_ch_time # by heng
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        # self.embed_fn = embed_fn # by heng
        self.zero_canonical = zero_canonical
        self._time, self._time_out, self._time_out_scale, self._time_out_rot, self._time_out_shs = self.create_time_net()

    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        # layers = [nn.Linear(self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 4), nn.Linear(self.W, 48)

    def query_time(self, new_pts, t, net, net_final, net_scale, net_rot, net_shs):
        h = torch.cat([new_pts, t], dim=-1)
        # h = t
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return net_final(h), net_scale(h), net_rot(h), net_shs(h)

    def forward(self, x, ts, iteration):
        input_pts = x

        # print ('iteration: ', iteration) # by heng

        input_pts = self.embed_fn(input_pts) # by heng
        ts = self.embedtime_fn(ts) # by heng

        # if iteration < 30000:
        #     noise = torch.randn(1).item() * 0.1 * (1 - iteration/20000)
        #     ts = ts + noise

        assert len(torch.unique(ts[:, :1])) == 1, "Only accepts all points from same time"
        cur_time = ts[0, 0]

        dx, dx_scale, dx_rot, mlp_shs = self.query_time(input_pts, ts, self._time, self._time_out, self._time_out_scale, self._time_out_rot, self._time_out_shs)

        # if cur_time == 0. or iteration < 3000: # by heng
        if iteration < 3000: #20010: #3000: #8000: # by heng
        # if False:
            dx = torch.zeros_like(input_pts[:, :3])
            dx_scale = torch.zeros_like(input_pts[:, :3])
            dx_rot = torch.zeros(input_pts.shape[0], 4).to(input_pts.device)
            mlp_shs = torch.zeros(input_pts.shape[0], 48).to(input_pts.device)
        # else:
            # dx, dx_scale, dx_rot, mlp_shs = self.query_time(input_pts, ts, self._time, self._time_out, self._time_out_scale, self._time_out_rot, self._time_out_shs)
        return dx, dx_scale, dx_rot, mlp_shs


## fw bw
# class DirectTemporalNeRF(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
#                  use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True):
#         super(DirectTemporalNeRF, self).__init__()


#         # self.embed_fn, self.input_ch = get_embedder(10, 3, 0)
#         # self.embedtime_fn, self.input_ch_time = get_embedder(10, 1, 0)

#         self.D = D
#         self.W = W
#         self.input_ch = input_ch   # by heng
#         self.input_ch_views = input_ch_views
#         self.input_ch_time = input_ch_time # by heng
#         self.skips = skips
#         self.use_viewdirs = use_viewdirs
#         self.memory = memory
#         self.embed_fn = embed_fn # by heng
#         self.zero_canonical = zero_canonical
#         self._time, self._time_out, self._fw, self._bw = self.create_time_net()

#     def create_time_net(self):
#         layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
#         for i in range(self.D - 1):
#             if i in self.memory:
#                 raise NotImplementedError
#             else:
#                 layer = nn.Linear

#             in_channels = self.W
#             if i in self.skips:
#                 in_channels += self.input_ch

#             layers += [layer(in_channels, self.W)]
#         return nn.ModuleList(layers), nn.Linear(self.W, 3), nn.Linear(self.W, 3), nn.Linear(self.W, 3)

#     def query_time(self, new_pts, t, net, net_final, net_fw, net_bw):
#         h = torch.cat([new_pts, t], dim=-1)
#         for i, l in enumerate(net):
#             h = net[i](h)
#             h = F.relu(h)
#             if i in self.skips:
#                 h = torch.cat([new_pts, h], -1)

#         return net_final(h), net_fw(h), net_bw(h)

#     def forward(self, x, ts, iteration):
#         input_pts = x

#         # print ('iteration: ', iteration) # by heng

#         # input_pts = self.embed_fn(input_pts)
#         # ts = self.embedtime_fn(ts)

#         assert len(torch.unique(ts[:, :1])) == 1, "Only accepts all points from same time"
#         cur_time = ts[0, 0]
#         # if cur_time == 0. : #or iteration < 3000: # by heng
#         if iteration < 3000: # by heng
#         # if False:
#             dx = torch.zeros_like(input_pts[:, :3])
#             d_fw = d_bw = dx
#         else:
#             dx, d_fw, d_bw = self.query_time(input_pts, ts, self._time, self._time_out, self._fw, self._bw)
#         return dx, d_fw, d_bw
    

class DirectTemporalNeRF_scaling(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3 + 3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True):
        super(DirectTemporalNeRF_scaling, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        self.embed_fn = embed_fn
        self.zero_canonical = zero_canonical
        self._time, self._time_out = self.create_time_net()

    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3)

    def query_time(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return net_final(h)

    def forward(self, x, ts, iteration):
        input_pts = x

        assert len(torch.unique(ts[:, :1])) == 1, "Only accepts all points from same time"
        cur_time = ts[0, 0]
        # if cur_time == 0. : #or iteration < 3000: # by heng
        if iteration < 3000: # by heng
        # if False:
            dx = torch.zeros_like(input_pts[:, :3])
        else:
            dx = self.query_time(input_pts, ts, self._time, self._time_out)
        return dx


class DirectTemporalNeRF_rot(nn.Module):
    def __init__(self, D=3, W=256, input_ch=4 + 3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True):
        super(DirectTemporalNeRF_rot, self).__init__()

        # self.embed_fn, self.input_ch = get_embedder(10, input_ch, 0)
        self.embedtime_fn, self.input_ch_time = get_embedder(10, input_ch_time, 0)

        self.D = D
        self.W = W
        self.input_ch = input_ch # by heng
        self.input_ch_views = input_ch_views
        # self.input_ch_time = input_ch_time # by heng
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        self.embed_fn = embed_fn # by heng
        self.zero_canonical = zero_canonical
        self._time, self._time_out = self.create_time_net()

    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 4)

    def query_time(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return net_final(h)

    def forward(self, x, ts, iteration):
        input_pts = x

        # input_pts = self.embed_fn(input_pts) # by heng
        ts = self.embedtime_fn(ts) # by heng

        assert len(torch.unique(ts[:, :1])) == 1, "Only accepts all points from same time"
        cur_time = ts[0, 0]
        # if cur_time == 0. or iteration < 3000: # by heng
        if iteration < 3000: # by heng
        # if False:
            # dx = torch.zeros_like(input_pts[:, :4])
            dx = torch.zeros(input_pts.shape[0], 4).to(input_pts.device)

        else:
            dx = self.query_time(input_pts, ts, self._time, self._time_out)
        return dx


class DirectTemporalNeRF_opacitymask(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True):
        super(DirectTemporalNeRF_opacitymask, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        self.embed_fn = embed_fn
        self.zero_canonical = zero_canonical
        self._time, self._time_out = self.create_time_net()

    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 1)

    def query_time(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return net_final(h)

    def forward(self, x, ts, iteration):
        input_pts = x

        assert len(torch.unique(ts[:, :1])) == 1, "Only accepts all points from same time"
        cur_time = ts[0, 0]
        # if cur_time == 0. : #or iteration < 3000: # by heng
        if iteration < 3000: # by heng
        # if False:
            dx = torch.ones_like(input_pts[:, :1])
        else:
            dx = self.query_time(input_pts, ts, self._time, self._time_out)
            dx = F.sigmoid(dx)
        return dx


class DirectTemporalNeRF_shs(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True):
        super(DirectTemporalNeRF_shs, self).__init__()

        # self.embed_fn, self.input_ch = get_embedder(10, 3, 0)
        # self.embedtime_fn, self.input_ch_time = get_embedder(10, 1, 0)

        self.D = D
        self.W = W
        self.input_ch = input_ch   # by heng
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time # by heng
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        self.embed_fn = embed_fn # by heng
        self.zero_canonical = zero_canonical
        self._time, self._time_out = self.create_time_net()

    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3 * 16) # by heng outdim for shs 16*3

    def query_time(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return net_final(h)

    def forward(self, x, ts, iteration):
        input_pts = x

        # print ('iteration: ', iteration) # by heng

        # input_pts = self.embed_fn(input_pts)
        # ts = self.embedtime_fn(ts)


        assert len(torch.unique(ts[:, :1])) == 1, "Only accepts all points from same time"
        cur_time = ts[0, 0]

        # if cur_time == 0. : #or iteration < 3000: # by heng
        if iteration < 3000: # by heng
        # if False:
            # dx = torch.zeros_like(input_pts[:, :3])
            dx = torch.zeros((input_pts.shape[0], 3*16), device=input_pts.device)
        else:
            dx = self.query_time(input_pts, ts, self._time, self._time_out)

        # dx = self.query_time(input_pts, ts, self._time, self._time_out)
        dx = dx.reshape(-1, 16, 3)
        return dx

 # by heng end

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # # by heng normalizing flow
        # def subnet_fc(dims_in, dims_out):
        #     return nn.Sequential(nn.Linear(dims_in, 512), nn.ReLU(),
        #                         nn.Linear(512,  dims_out))

        # # a simple chain of operations is collected by ReversibleSequential
        # self.inn = Ff.SequenceINN(3)
        # for k in range(8):
        #     self.inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
        # # by heng end

        self.offset_model = DirectTemporalNeRF() # by heng
        self.offset_model_rot = DirectTemporalNeRF_rot() # by heng
        self.offset_model_scaling = DirectTemporalNeRF_scaling()
        self.opacity_mask =DirectTemporalNeRF_opacitymask() # by heng
        self.shs_model =DirectTemporalNeRF_shs() # by heng

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

        # # remove some param by heng
        # opt_dict['param_groups'] = [group for group in opt_dict['param_groups'] if group.get('name') not in ['xyz', 'scaling', 'rotation', 'offset_model', 'offset_model_rot', 'opacity_mask', 'shs_model']]
        # opt_dict['param_groups'] = [group for group in opt_dict['param_groups'] if group.get('name') not in ['xyz', 'rotation']]
        # opt_dict['param_groups'] = [group for group in opt_dict['param_groups'] if group.get('name') in ['f_dc', 'f_rest', 'opacity', 'scaling']]

        self.optimizer.load_state_dict(opt_dict)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                state[k] = v.cuda()


    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    def get_scaling_withOffset(self, pts, time, iteration): # by heng
        # scaling_offset = self.get_offset_scaling(self._rotation, time, iteration) # by heng
        scaling_offset = self.get_offset_scaling(pts, time, iteration) # by heng

        return self.scaling_activation(self._scaling + scaling_offset)

    @property
    def get_rotation_ori(self): # by heng
        return self.rotation_activation(self._rotation)
        
    # @property
    def get_rotation(self, pts, time, iteration): # by heng
        # rot_offset = self.get_offset_rot(self._rotation, time, iteration) # by heng
        rot_offset = self.get_offset_rot(pts, time, iteration) # by heng

        return self.rotation_activation(self._rotation + rot_offset), rot_offset


    @property
    def get_xyz(self):
        return self._xyz

    # def get_xyz_all(self, pts, time, iteration):
    #     xyz_offset = self.get_offset(pts, time, iteration)
    #     return self._xyz + xyz_offset, self._xyz, xyz_offset

    def get_xyz_all(self, pts, time, iteration):
        xyz_offset, scale_offset, rot_offset, mlp_shs = self.get_offset(pts, time, iteration)
        return self._xyz + xyz_offset, self._xyz, xyz_offset, scale_offset, rot_offset, mlp_shs


    # by heng
    def get_offset(self, pts, time, iteration):
        offset_model  = self.offset_model.to(pts.device)
        return offset_model(pts, time, iteration)

    def get_offset_scaling(self, pts, time, iteration):
        offset_model_scaling  = self.offset_model_scaling.to(pts.device)
        return offset_model_scaling(pts, time, iteration)

    def get_offset_rot(self, pts, time, iteration):
        offset_model_rot  = self.offset_model_rot.to(pts.device)
        return offset_model_rot(pts, time, iteration)

    def get_opacity_mask(self, pts, time, iteration):
        opacity_mask  = self.opacity_mask.to(pts.device)
        return opacity_mask(pts, time, iteration)

    def get_shs(self, pts, time, iteration):
        shs_model  = self.shs_model.to(pts.device)
        return shs_model(pts, time, iteration)

    # by heng end

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)


    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_3vec = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            ##### {'params': self.inn.parameters(), 'lr': training_args.rotation_lr, "name": "inn"}, # by heng normalizing flow
            {'params': self.offset_model.parameters(), 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "offset_model"}, # by heng
            # {'params': self.offset_model_rot.parameters(), 'lr': training_args.rotation_lr, "name": "offset_model_rot"}, # by heng
            # {'params': self.offset_model_scaling.parameters(), 'lr': training_args.scaling_lr, "name": "offset_model_scaling"}, # by heng
            # {'params': self.opacity_mask.parameters(), 'lr': training_args.opacity_lr, "name": "opacity_mask"}, # by heng
            # {'params': self.shs_model.parameters(), 'lr': training_args.feature_lr, "name": "shs_model"}, # by heng

            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.offset_scheduler_args = get_expon_lr_func(lr_init=8e-4,
                                                    lr_final=1.6e-6,
                                                    max_steps=training_args.position_lr_max_steps)

    # def update_learning_rate(self, iteration):
    #     ''' Learning rate scheduling per step '''
    #     for param_group in self.optimizer.param_groups:
    #         if param_group["name"] == "xyz":
    #             lr = self.xyz_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #             return lr

    # by heng 
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        # if iteration < 30000:
        for param_group in self.optimizer.param_groups:
            # print ('lr: ', param_group["name"], param_group['lr'])
            if param_group["name"] in ["xyz"]:
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] in ["offset_model", "offset_model_rot"]:
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr                
            
    
    # by heng end

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        path_name = os.path.dirname(path) # by heng
        torch.save(self.offset_model.state_dict(), f'{path_name}/offset_model.pth') # by heng
        torch.save(self.offset_model_rot.state_dict(), f'{path_name}/offset_model_rot.pth') # by heng
        torch.save(self.offset_model_scaling.state_dict(), f'{path_name}/offset_model_scaling.pth') # by heng
        torch.save(self.opacity_mask.state_dict(), f'{path_name}/opacity_mask.pth') # by heng
        torch.save(self.shs_model.state_dict(), f'{path_name}/shs_model.pth') # by heng


    def save_ply_t(self, path, xyz, opacities, rotation): # by heng 
        mkdir_p(os.path.dirname(path))

        xyz = xyz.detach().cpu().numpy() #self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacities.detach().cpu().numpy() #self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy() #self._rotation.detach().cpu().numpy()

        # xyz = self._xyz.detach().cpu().numpy()
        # normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self._opacity.detach().cpu().numpy()
        # scale = self._scaling.detach().cpu().numpy()
        # rotation = self._rotation.detach().cpu().numpy()


        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        # by heng
        self.offset_model = DirectTemporalNeRF() 
        self.offset_model_rot = DirectTemporalNeRF_rot() 
        self.offset_model_scaling = DirectTemporalNeRF_scaling()
        self.opacity_mask = DirectTemporalNeRF_opacitymask()
        self.shs_model = DirectTemporalNeRF_shs()
        path_name = os.path.dirname(path)
        self.offset_model.load_state_dict(torch.load(f'{path_name}/offset_model.pth'))
        self.offset_model_rot.load_state_dict(torch.load(f'{path_name}/offset_model_rot.pth'))
        self.offset_model_scaling.load_state_dict(torch.load(f'{path_name}/offset_model_scaling.pth'))
        self.opacity_mask.load_state_dict(torch.load(f'{path_name}/opacity_mask.pth'))
        self.shs_model.load_state_dict(torch.load(f'{path_name}/shs_model.pth'))
        self.offset_model.eval()
        self.offset_model_rot.eval()
        self.offset_model_scaling.eval()
        self.opacity_mask.eval()
        self.shs_model.eval()
        # by heng end

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:

            if len(group["params"]) != 1: # by heng normalizing flow
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_3vec = self.xyz_gradient_accum_3vec[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            
            if len(group["params"]) != 1: # by heng normalizing flow
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_3vec = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_split_4offset(self, grads, grad_threshold, scene_extent, offset_mask, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition

        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        padded_offset_mask = torch.zeros((n_init_points), device="cuda") # by heng
        padded_offset_mask[:grads.shape[0]] = offset_mask # by heng

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, padded_offset_mask) # by heng 
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)


    def densify_and_clone_4offset(self, grads, grad_threshold, scene_extent, offset_mask):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, offset_mask) # by heng 
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify_and_prune_4offset(self, max_grad, min_opacity, extent, max_screen_size, offset_mask): # by heng
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone_4offset(grads, max_grad, extent, offset_mask)
        self.densify_and_split_4offset(grads, max_grad, extent, offset_mask)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):

        self.xyz_gradient_accum_3vec[update_filter] += viewspace_point_tensor.grad[update_filter]

        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
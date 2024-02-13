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

import os
import numpy as np
import torch
from torch import nn
from random import randint, shuffle
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
from plyfile import PlyData, PlyElement
np.random.seed()

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def kabsch_rigid_transform(P, Q):
    # Kabsch algorithm to find the best rigid transform between two sets of points
    # Assumes P and Q are both (num_points, 3) tensors
    H = P.t().mm(Q)
    U, _, Vt = torch.svd(H)
    R = Vt.t().mm(U.t())
    return R

def get_nearest_points(locations, k=20):
    """
    Get the nearest 'k' points for each location.

    Args:
    - locations (torch.Tensor): A tensor of shape (N, 3) representing 3D points.
    - k (int): Number of nearest neighbors to retrieve.

    Returns:
    - torch.Tensor: Indices of the nearest 'k' points for each location. Shape will be (N, k).
    """
    # Compute pairwise squared distances
    dot_product = torch.mm(locations, locations.t())
    square_norms = dot_product.diag()
    distances = square_norms.unsqueeze(1) - 2.0 * dot_product + square_norms.unsqueeze(0)
    distances.fill_diagonal_(float('inf'))  # Fill diagonals with inf so that a point doesn't consider itself

    # Get the indices of the nearest 'k' points. Exclude the point itself.
    _, indices = distances.sort(dim=1)
    return indices[:, :k]

def quat_to_rotm(quaternion):
    """Convert a batch of quaternions to rotation matrices.
    Args:
        quaternion (torch.Tensor): tensor of quaternions with shape [N, 4], with each row as [qw, qx, qy, qz].
    Returns:
        torch.Tensor: tensor of rotation matrices with shape [N, 3, 3].
    """
    qw, qx, qy, qz = quaternion.split(1, dim=1)

    r00 = 1 - 2 * (qy**2 + qz**2)
    r01 = 2 * (qx*qy - qw*qz)
    r02 = 2 * (qx*qz + qw*qy)

    r10 = 2 * (qx*qy + qw*qz)
    r11 = 1 - 2 * (qx**2 + qz**2)
    r12 = 2 * (qy*qz - qw*qx)

    r20 = 2 * (qx*qz - qw*qy)
    r21 = 2 * (qy*qz + qw*qx)
    r22 = 1 - 2 * (qx**2 + qy**2)

    rotation_matrix = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=2).view(-1, 3, 3)
    return rotation_matrix



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        # (model_params, first_iter) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt)

        path_name = os.path.dirname(checkpoint)
        ckpt_id = checkpoint.split('/')[-1].split('.')[0].split('_')[-1]

        gaussians.offset_model.load_state_dict(torch.load(f'{path_name}/offset_model_{ckpt_id}.pth', map_location="cuda:0"))
        gaussians.offset_model_rot.load_state_dict(torch.load(f'{path_name}/offset_model_rot_{ckpt_id}.pth', map_location="cuda:0"))
        gaussians.offset_model_scaling.load_state_dict(torch.load(f'{path_name}/offset_model_scaling_{ckpt_id}.pth', map_location="cuda:0"))
        gaussians.opacity_mask.load_state_dict(torch.load(f'{path_name}/opacity_mask_{ckpt_id}.pth', map_location="cuda:0"))
        gaussians.shs_model.load_state_dict(torch.load(f'{path_name}/shs_model_{ckpt_id}.pth', map_location="cuda:0"))

        (model_params, first_iter) = torch.load(checkpoint,  map_location='cuda:0')
        gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    idx_stack = None
    nearest_indices = None
    offset_mask = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):    

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            # viewpoint_stack = sorted(viewpoint_stack, key=lambda cam: cam.time) # by heng
            viewpoint_cam = None
        viewpoint_cam_prev = viewpoint_cam
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        # viewpoint_cam = viewpoint_stack[10]# by heng 
        
        # if viewpoint_cam_prev is not None:
        #     print ('here')
        #     print (viewpoint_cam_prev.time)
        #     print (viewpoint_cam.time)
        #     raise

        # # get two adjacent cams
        # # if not viewpoint_stack:
        # #     viewpoint_stack = scene.getTrainCameras().copy()
        # #     viewpoint_stack = sorted(viewpoint_stack, key=lambda cam: cam.time) # by heng 

        # if not idx_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()
        #     viewpoint_stack = sorted(viewpoint_stack, key=lambda cam: cam.time) # by heng 

        #     # idx_stack = list(np.random.permutation(len(viewpoint_stack)))
        #     idx_stack = list(range(len(viewpoint_stack)))
        #     # shuffle(idx_stack)

        # rand_idx = idx_stack.pop(randint(0, len(idx_stack)-1))
        # # rand_idx = randint(0, len(viewpoint_stack)-1)

        # viewpoint_cam = viewpoint_stack[rand_idx]
        # if rand_idx == 0:
        #     viewpoint_cam_prev = None
        #     # viewpoint_cam_next = viewpoint_stack[rand_idx+1]
        # # elif rand_idx == len(viewpoint_stack)-1:
        # #     viewpoint_cam_prev = viewpoint_stack[rand_idx-1]
        # #     viewpoint_cam_next = None
        # else:
        #     viewpoint_cam_prev = viewpoint_stack[rand_idx-1]
        #     # viewpoint_cam_next = viewpoint_stack[rand_idx+1]


        # # get three cams
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()
        #     viewpoint_stack = sorted(viewpoint_stack, key=lambda cam: cam.time) # by heng 

        # if not idx_stack:
        #     idx_stack = list(np.random.permutation(len(viewpoint_stack)))

        # rand_idx = idx_stack.pop()
        # # rand_idx = randint(0, len(viewpoint_stack)-1)

        # viewpoint_cam = viewpoint_stack[rand_idx]
        # if rand_idx == 0:
        #     viewpoint_cam_prev = None
        #     viewpoint_cam_next = viewpoint_stack[rand_idx+1]
        # elif rand_idx == len(viewpoint_stack)-1:
        #     viewpoint_cam_prev = viewpoint_stack[rand_idx-1]
        #     viewpoint_cam_next = None
        # else:
        #     viewpoint_cam_prev = viewpoint_stack[rand_idx-1]
        #     viewpoint_cam_next = viewpoint_stack[rand_idx+1]



        # # by heng
        # if iteration > 3000:
        #     viewpoint_stack = scene.getTrainCameras().copy()
        #     viewpoint_stack = sorted(viewpoint_stack, key=lambda cam: cam.time) # by heng 
        #     viewpoint_cam = viewpoint_stack[25]

        # idx = randint(0, len(viewpoint_stack)-1)
        # idx_1 = idx - 1 if idx == len(viewpoint_stack) -1 else idx + 1
        # viewpoint_cam = viewpoint_stack[idx]
        # viewpoint_cam_1 = viewpoint_stack[idx_1]   
        # # by heng end

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, iteration)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        means3D_xyz_ori = render_pkg["means3D_ori"]

        means3D_offset = render_pkg["means3D_offset"]
        rot_offset = render_pkg["rot_offset"]

        # # by heng
        # rigid_loss = 0.
        # if iteration > 3000:
        #     # if iteration == 3001:
        #         # viewpoint_cam0 = viewpoint_stack[0]
        #         # render_pkg0 = render(viewpoint_cam0, gaussians, pipe, background, iteration)
        #         # means3D0 = render_pkg0["means3D"]
        #         # nearest_indices = get_nearest_points(means3D0.cpu(), 20).to(render_pkg0["rotations"].device)
        #         # lambda_w = 200

        #         # N = len(means3D0)
        #         # # Gather nearest points using indexing
        #         # means3D0_near = torch.index_select(means3D0, 0, nearest_indices.view(-1)).view(N, -1, 3)
        #         # tmp = means3D0_near - means3D0[:, None]
        #         # squared_norms = (tmp * tmp).sum(dim=-1)
        #         # wij = torch.exp(-lambda_w * squared_norms).cpu().detach().numpy()
        #         # wij = torch.from_numpy(wij).to(nearest_indices.device)

        #     means3D, rotations = render_pkg["means3D"], render_pkg["rotations"]
        #     render_pkg_1 = render(viewpoint_cam_1, gaussians, pipe, background, iteration)
        #     image_1 = render_pkg_1["render"]
        #     means3D_1, rotations_1 = render_pkg_1["means3D"], render_pkg_1["rotations"]
        #     rotms = quat_to_rotm(rotations)
        #     rotms_1 = quat_to_rotm(rotations_1)


        #     nearest_indices = get_nearest_points(means3D.cpu(), 20).to(rotations.device)
        #     lambda_w = 20
        #     N = len(means3D)
        #     # Gather nearest points using indexing
        #     means3D_near = torch.index_select(means3D, 0, nearest_indices.view(-1)).view(N, -1, 3)
        #     tmp = means3D_near - means3D[:, None]
        #     squared_norms = (tmp * tmp).sum(dim=-1)
        #     wij = torch.exp(-lambda_w * squared_norms).cpu().detach().numpy()
        #     wij = torch.from_numpy(wij).to(nearest_indices.device)


        #     # rigid_loss = []
        #     # for point_i in range(len(means3D)):
        #     #     print (point_i)
        #     #     nearest_indices_i = nearest_indices[point_i]
        #     #     means3D0_i = means3D0[point_i]
        #     #     means3D0_i_near = means3D0[nearest_indices_i]
        #     #     tmp = means3D0_i_near - means3D0_i
        #     #     squared_norms = (tmp * tmp).sum(dim=-1)
        #     #     wij = torch.exp(-lambda_w * squared_norms)

        #     #     means3D_i = means3D[point_i]
        #     #     means3D_1_i = means3D_1[point_i]
        #     #     rotms_i = rotms[point_i]
        #     #     rotms_1_i = rotms_1[point_i]
        #     #     rotations_i = rotations[point_i]
        #     #     rotations_1_i = rotations_1[point_i]

        #     #     tmp1 = (means3D[nearest_indices_i] - means3D_i) - torch.matmul(means3D_1[nearest_indices_i] - means3D_1_i, (rotms_i * rotms_1_i.transpose(0, 1)).t())
        #     #     rigid_loss_i = wij*torch.norm(tmp1, dim=-1)
        #     #     rigid_loss.append(rigid_loss_i.mean())

        #     # rigid_loss = torch.mean(torch.stack(rigid_loss))

        #     # Computing tmp1 for all points in batch

        #     diff_means3D = torch.index_select(means3D, 0, nearest_indices.view(-1)).view(N, -1, 3) - means3D[:, None]
        #     diff_means3D_1 = torch.index_select(means3D_1, 0, nearest_indices.view(-1)).view(N, -1, 3) - means3D_1[:, None]
        #     rot_mult = torch.matmul(rotms, rotms_1.transpose(1, 2))
        #     tmp1 = diff_means3D - torch.matmul(diff_means3D_1, rot_mult.transpose(1, 2))

        #     rigid_loss = (wij * torch.norm(tmp1, dim=-1)).mean(dim=-1)
        #     rigid_loss = rigid_loss.mean()

        #     print(rigid_loss)

        # # by heng end

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        # Ll1 = l2_loss(image, gt_image)
        # Ll1 = 0.5 * l1_loss(image, gt_image) + 0.5 * l2_loss(image, gt_image)
        # Ll1 = l1_loss(image, gt_image) + l2_loss(image, gt_image)

        # # offset norm loss by heng
        L_offsetnrom = torch.norm(means3D_offset, dim=-1).mean()
        if iteration % 500 == 0:
            print ('Ll1: ', Ll1, 'L_offsetnrom: ', L_offsetnrom)
        Ll1 = Ll1 + 0.1 * L_offsetnrom # 0.01 by heng 0.001 for jumping jacks
        ## # offset norm loss by heng end

        # if viewpoint_cam_prev is not None and iteration > opt.densify_until_iter: #30000, 15_000
        #     offset_mask_0 = torch.where(torch.norm(means3D_offset, dim=-1) < 0.1, True, False) # 0.1 by heng
        #     means3D_offset_0 = means3D_offset[offset_mask_0]
        #     L_offsetnrom_0 = torch.norm(means3D_offset_0, dim=-1).mean()
        #     if iteration % 500 == 0:
        #         print ('Ll1: ', Ll1, 'L_offsetnrom_0: ', L_offsetnrom_0)
        #     Ll1 = Ll1 + L_offsetnrom_0 

        # # # rot_offset norm loss by heng
        # L_rot_offset_norm = torch.norm(rot_offset, dim=-1).mean()
        # if iteration % 500 == 0:
        #     print ('Ll1: ', Ll1, 'L_rot_offset_norm: ', L_rot_offset_norm)
        # Ll1 = Ll1 + 0.0001 * L_rot_offset_norm # 0.01 by heng
        # ## # rot_offset norm loss by heng end


        # offset_mask = torch.where(torch.norm(means3D_offset, dim=-1) >= 0.1, True, False) # 0.1 by heng

        # # tv loss by heng
        # if len(means3D_offset[offset_mask]) > 500 and iteration > 30000: #100: by heng
        #     means3D, rotations = render_pkg["means3D_ori"], render_pkg["rotations"]

        #     means3D_0 = means3D[offset_mask]
        #     means3D_offset_0 = means3D_offset[offset_mask]
        #     N = len(means3D_0)
        #     nearest_indices = get_nearest_points(means3D_0.cpu(), 1).to(rotations.device)

        #     # means3D_0_near = torch.index_select(means3D_0, 0, nearest_indices.view(-1)).view(N, 3)

        #     means3D_offset_0_near = torch.index_select(means3D_offset_0, 0, nearest_indices.view(-1)).view(N, 3)
        #     l_tv= torch.sqrt(((means3D_offset_0 - means3D_offset_0_near) ** 2).sum(dim=-1)).mean()
        #     if iteration % 200 == 0:
        #         print ('Ll1: ', Ll1, 'l_tv: ', l_tv)
        #     # Ll1 = Ll1 + 0.001 * l_tv
        # # tv loss by heng end


        # # loss for nearest point dist in prev and cur t by heng
        # # if viewpoint_cam_prev is not None and len(means3D_offset[offset_mask]) > 500 and iteration > 30000:
        # if viewpoint_cam_prev is not None and iteration > opt.densify_until_iter: #30000, 15_000
        # # if viewpoint_cam_prev is not None and ((iteration >= opt.densify_until_iter and iteration % 10000 >= 5000)): # or iteration >= 50000):

        #     if offset_mask is None:
        #         offset_mask = torch.where(torch.norm(means3D_offset, dim=-1) >= 0.1, True, False) # 0.1 by heng
        #     render_pkg_prev = render(viewpoint_cam_prev, gaussians, pipe, background, iteration)
        #     means3D_offset_prev = render_pkg_prev["means3D_offset"]
        #     means3D_prev = render_pkg_prev["means3D"]
        #     means3D, rotations = render_pkg["means3D"], render_pkg["rotations"]
        #     means3D_0 = means3D[offset_mask]
        #     means3D_prev_0 = means3D_prev[offset_mask]
        #     N = len(means3D_0)

        #     # nearest one points
        #     if nearest_indices is None:
        #         # nearest_indices = get_nearest_points(means3D_0.cpu(), 1).to(rotations.device)
        #         means3D_ori = render_pkg["means3D_ori"][offset_mask]
        #         nearest_indices = get_nearest_points(means3D_ori.cpu(), 1).to(rotations.device) # 1 by heng

        #     means3D_0_near = torch.index_select(means3D_0, 0, nearest_indices.view(-1)).view(N, 3)
        #     means3D_prev_0_near = torch.index_select(means3D_prev_0, 0, nearest_indices.view(-1)).view(N, 3)
        #     diff = means3D_0 - means3D_0_near
        #     l2_dist = torch.norm(diff, dim=1)
        #     diff_prev = means3D_prev_0 - means3D_prev_0_near
        #     l2_dist_prev = torch.norm(diff_prev, dim=1)
        #     Ll1_reg = l1_loss(l2_dist, l2_dist_prev)
        #     if iteration % 500 == 0:
        #         print ('Ll1: ', Ll1, 'Ll1_reg: ', Ll1_reg)
        #     Ll1 = Ll1 + 1 * Ll1_reg

        #     # # nearest N points
        #     # if nearest_indices is None:
        #     #     # nearest_indices = get_nearest_points(means3D_0.cpu(), 1).to(rotations.device)
        #     #     means3D_ori = render_pkg["means3D_ori"][offset_mask]
        #     #     nearest_indices = get_nearest_points(means3D_ori.cpu(), 20).to(rotations.device) # N = 20 by heng

        #     # means3D_0_near = torch.index_select(means3D_0, 0, nearest_indices.view(-1)).view(N, -1, 3)
        #     # means3D_prev_0_near = torch.index_select(means3D_prev_0, 0, nearest_indices.view(-1)).view(N, -1, 3)

        #     # diff = means3D_0_near - means3D_0[:, None, :]
        #     # l2_dist = torch.norm(diff, dim=-1)
        #     # diff_prev = means3D_prev_0_near - means3D_prev_0[:, None, :]
        #     # l2_dist_prev = torch.norm(diff_prev, dim=-1)

        #     # Ll1_reg = torch.abs((l2_dist - l2_dist_prev)).sum(dim=-1).mean() #l1_loss(l2_dist, l2_dist_prev)
        #     # if iteration % 500 == 0:
        #     #     print ('Ll1: ', Ll1, 'Ll1_reg: ', Ll1_reg)
        #     # Ll1 = Ll1 + 1 * Ll1_reg


        #     # # # rot loss by heng
        #     # rotations, rotations_prev = render_pkg["rotations"][offset_mask], render_pkg_prev["rotations"][offset_mask]
        #     # rotations_near = torch.index_select(rotations, 0, nearest_indices.view(-1)).view(N, 4)
        #     # rotations_prev_near = torch.index_select(rotations_prev, 0, nearest_indices.view(-1)).view(N, 4)
        #     # diff_rot = rotations - rotations_near
        #     # l2_dist_rot = torch.norm(diff_rot, dim=1)
        #     # diff_prev_rot = rotations_prev - rotations_prev_near
        #     # l2_dist_prev_rot = torch.norm(diff_prev_rot, dim=1)
        #     # Ll1_reg_rot = l1_loss(l2_dist_rot, l2_dist_prev_rot)
        #     # if iteration % 500 == 0:
        #     #     print ('Ll1: ', Ll1, 'Ll1_reg_rot: ', Ll1_reg_rot)
        #     #     print ('l2_dist_rot: ', l2_dist_rot.mean(), 'l2_dist_prev_rot: ', l2_dist_prev_rot.mean())
        #     # Ll1 = Ll1 + 1 * Ll1_reg_rot
        #     # # # rot loss by heng end



        # #     # # rigid loss by heng
        # #     # rotations, rotations_prev = render_pkg["rotations"][offset_mask], render_pkg_prev["rotations"][offset_mask]
        # #     # rotms = quat_to_rotm(rotations)
        # #     # rotms_prev = quat_to_rotm(rotations_prev)
        # #     # diff_means3D = diff
        # #     # diff_means3D_prev = diff_prev
        # #     # rot_mult = torch.matmul(rotms, rotms_prev.transpose(1, 2))
        # #     # rigid_loss_tmp1 = diff_means3D - torch.matmul(diff_means3D_prev, rot_mult.transpose(1, 2))
        # #     # rigid_loss = torch.norm(rigid_loss_tmp1, dim=-1)
        # #     # rigid_loss = rigid_loss.mean()
        # #     # if iteration % 500 == 0:
        # #     #     print ('Ll1: ', Ll1, 'rigid_loss: ', rigid_loss)
        # #     # Ll1 = Ll1 + 1 * rigid_loss

        # # # loss for nearest point dist in prev and cur t by heng end


        # if viewpoint_cam_prev is None:
        #     render_pkg_next = render(viewpoint_cam_next, gaussians, pipe, background, iteration)
        #     means3D_offset_next = render_pkg_next["means3D_offset"]
        #     # Ll1_reg = l1_loss(means3D_offset, means3D_offset_next)

        # elif viewpoint_cam_next is None:
        #     render_pkg_prev = render(viewpoint_cam_prev, gaussians, pipe, background, iteration)
        #     means3D_offset_prev = render_pkg_prev["means3D_offset"]
        #     # Ll1_reg = l1_loss(means3D_offset, means3D_offset_prev)
        # else:
        #     render_pkg_prev = render(viewpoint_cam_prev, gaussians, pipe, background, iteration)
        #     means3D_offset_prev = render_pkg_prev["means3D_offset"]
        #     render_pkg_next = render(viewpoint_cam_next, gaussians, pipe, background, iteration)
        #     means3D_offset_next = render_pkg_next["means3D_offset"]
        #     # Ll1_reg = l1_loss(means3D_offset_next + means3D_offset_prev, 2 * means3D_offset)


        # if iteration % 1000 == 0:
        #     print ('number of movement: ', len(means3D_offset[offset_mask]))
        #     print (torch.norm(means3D_offset, dim=-1).max(), torch.norm(means3D_offset, dim=-1).min())

        # if len(means3D_offset[offset_mask]) > 100:

        #     if viewpoint_cam_prev is not None:
        #         render_pkg_prev = render(viewpoint_cam_prev, gaussians, pipe, background, iteration)
        #         means3D_offset_prev = render_pkg_prev["means3D_offset"]
        #         P = means3D_offset[offset_mask]
        #         Q = means3D_offset_prev[offset_mask]
        #         R = kabsch_rigid_transform(P, Q)
        #         transformed_P = P.mm(R.t())
        #         Ll1_reg_prev = nn.functional.mse_loss(transformed_P, Q)
        #         # print ('Ll1: ', Ll1, 'Ll1_reg_prev: ', Ll1_reg_prev)
        #         Ll1 = Ll1 + 0.01 * Ll1_reg_prev

        #     if viewpoint_cam_next is not None:
        #         render_pkg_next = render(viewpoint_cam_next, gaussians, pipe, background, iteration)
        #         means3D_offset_next = render_pkg_next["means3D_offset"]
        #         P = means3D_offset[offset_mask]
        #         Q = means3D_offset_next[offset_mask]
        #         R = kabsch_rigid_transform(P, Q)
        #         transformed_P = P.mm(R.t())
        #         Ll1_reg_next = nn.functional.mse_loss(transformed_P, Q)
        #         # print ('Ll1: ', Ll1, 'Ll1_reg_next: ', Ll1_reg_next)
        #         Ll1 = Ll1 + 0.01 * Ll1_reg_next


        # print ('Ll1: ', Ll1, 'Ll1_reg: ', Ll1_reg)

        # Ll1 = Ll1 + 0.1 * Ll1_reg

        # # by heng test
        # with torch.no_grad():
        #     offset_mask = torch.where(torch.norm(means3D_offset, dim=-1) >= 0.1, True, False)
        # if iteration > 10000:
        #     tmp = torch.abs((image - gt_image)).sum(0)
        #     mask = tmp > 0.5

        #     print (mask.shape)
        #     print (offset_mask.shape)
        #     raise

        #     Ll1 = (torch.abs((image - gt_image)) * mask[None, ...]).sum() / mask.sum()
        #     # print ('tmp : ', tmp.shape, tmp.max(), tmp.min())
        #     # torchvision.utils.save_image(tmp, "tmp.png")
        #     # print (Ll1)
        #     # raise
        # else:
        #     Ll1 = l1_loss(image, gt_image)
        # # by heng end


        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + rigid_loss # by heng
        # loss = l2_loss(image, gt_image) #(1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * l2_loss(image, gt_image) # by heng
        # loss = Ll1 # by heng

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, iteration))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # print ('viewspace_point_tensor: ', viewspace_point_tensor.grad.shape, viewspace_point_tensor.grad.max(), viewspace_point_tensor.grad.min())


            # # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold) # 0.005 by heng
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()



            # # debug by heng 
            # # if iteration > 31000:
            # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            # grads = gaussians.xyz_gradient_accum / gaussians.denom
            # grads[grads.isnan()] = 0.0
            # print (grads.shape, grads.max(), grads.min())

            # selected_pts_mask = torch.where(grads >= 0.0001, True, False).squeeze()
            # selected_pts_mask_0 = torch.where(torch.norm(means3D_offset, dim=-1) >= 0.5, True, False)
            # selected_pts_mask = torch.logical_and(selected_pts_mask, selected_pts_mask_0) # by heng 

            # print (len (grads), len(grads[selected_pts_mask]))
            # path = f'point_cloud_test.ply'
            # xyz=render_pkg["means3D"][selected_pts_mask]
            # opacities=render_pkg["opacities"][selected_pts_mask]
            # rotation=render_pkg["rotations"][selected_pts_mask]

            # xyz = xyz.detach().cpu().numpy() #self._xyz.detach().cpu().numpy()
            # normals = np.zeros_like(xyz)
            # f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            # f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            # opacities = opacities.detach().cpu().numpy() #self._opacity.detach().cpu().numpy()
            # scale = gaussians._scaling.detach().cpu().numpy()
            # rotation = rotation.detach().cpu().numpy() #self._rotation.detach().cpu().numpy()

            # dtype_full = [(attribute, 'f4') for attribute in gaussians.construct_list_of_attributes()]

            # elements = np.empty(xyz.shape[0], dtype=dtype_full)
            # attributes = np.concatenate((xyz, normals, f_dc[selected_pts_mask.cpu().numpy()], f_rest[selected_pts_mask.cpu().numpy()], opacities, scale[selected_pts_mask.cpu().numpy()], rotation), axis=1)
            # elements[:] = list(map(tuple, attributes))
            # el = PlyElement.describe(elements, 'vertex')
            # PlyData([el]).write(path)
            # raise
            # # debug by heng end 


            # Densification by heng 
            if iteration < opt.densify_until_iter:
            # if (iteration < opt.densify_until_iter or iteration % 10000 < 5000): #and iteration < 50000:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                # print (viewspace_point_tensor.grad)
                # raise

                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                min_opacity = 0.005 #0.005 # by heng 
                offset_mask = None
                nearest_indices = None

                if not iteration < opt.densify_until_iter:
    
                    grads = gaussians.xyz_gradient_accum / gaussians.denom
                    grads[grads.isnan()] = 0.0
                    opt.densify_grad_threshold = grads.max() * 0.5
                    min_opacity = min_opacity * 0.5

                # if iteration > 15000: # by heng
                #     # min_opacity = 0.002
                #     opt.densify_grad_threshold = 0.0001

                # if iteration > 15000: # by heng
                #     # min_opacity = 0.002
                #     opt.densify_grad_threshold = 0.00005

                # if iteration > 30000: # by heng
                #     # min_opacity = 0.002
                #     opt.densify_grad_threshold = 0.00001

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold) # 0.005 by heng
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    
            # by heng end 

            # # by heng tmp
            # if opt.densify_until_iter <= iteration < 30000 and len(means3D_offset[offset_mask]) > 100:
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         offset_mask = torch.where(torch.norm(means3D_offset, dim=-1) >= 0.3, True, False) # by heng

            #         gaussians.densify_and_prune_4offset(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, offset_mask)

            # if 30000 <= iteration < 40000 and len(means3D_offset[offset_mask]) > 100:

            #         gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #         gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    
            #         opt.densify_grad_threshold = 0.00005

            #         if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #             size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #             offset_mask = torch.where(torch.norm(means3D_offset, dim=-1) >= 0.5, True, False) # by heng

            #             gaussians.densify_and_prune_4offset(opt.densify_grad_threshold, 0.015, scene.cameras_extent, size_threshold, offset_mask)

            # #by heng tmo end



            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                # torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_" + str(iteration) + ".pth")


                path_name = scene.model_path + '/ckpt_save'
                os.makedirs(path_name, exist_ok=True)
                torch.save((gaussians.capture(), iteration), path_name + "/chkpnt_" + str(iteration) + ".pth")
                torch.save(gaussians.offset_model.state_dict(), f'{path_name}/offset_model_{iteration}.pth') # by heng
                torch.save(gaussians.offset_model_rot.state_dict(), f'{path_name}/offset_model_rot_{iteration}.pth') # by heng
                torch.save(gaussians.offset_model_scaling.state_dict(), f'{path_name}/offset_model_scaling_{iteration}.pth') # by heng
                torch.save(gaussians.opacity_mask.state_dict(), f'{path_name}/opacity_mask_{iteration}.pth') # by heng
                torch.save(gaussians.shs_model.state_dict(), f'{path_name}/shs_model_{iteration}.pth') # by heng


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6019) #6009
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    iter_list = [7_000, 15_000] + [i for i in range(20_000, 900_000 + 1, 10_000)]
    parser.add_argument("--test_iterations", nargs="+", type=int, default=iter_list)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=iter_list)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=iter_list)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

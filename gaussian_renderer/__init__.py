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
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import scene.rigid_body as rigid

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, iteration=0, scaling_modifier = 1.0, override_color = None, save_ply=False, control_time=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    
    # means3D_ori = means3D
    rotation_tmp = pc.get_rotation_ori
    scaling_tmp = pc.get_scaling
    means3D_rot = torch.cat((means3D, rotation_tmp), dim=1)
    means3D_scaling = torch.cat((means3D, scaling_tmp), dim=1)

    if control_time is not None:
        time = torch.full((means3D.size(0), 1), control_time, device=means3D.device) # by heng
    else:
        time = torch.full((means3D.size(0), 1), viewpoint_camera.time, device=means3D.device) # by heng
    # time = time * 0 #+ 0.6667 # by heng


    # # by heng
    # screenspace_points = torch.zeros_like(pc.get_xyz_all(means3D, time, iteration)[0], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0 # by heng 
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass
    # #by heng end


    means3D, means3D_ori, means3D_offset, scale_offset, rot_offset, mlp_shs = pc.get_xyz_all(means3D, time, iteration) # means3D, means3D_rot by heng
    # means3D, means3D_ori, means3D_offset = pc.get_xyz_all(means3D, time, iteration)
    # means3D_offset = pc.get_offset(means3D, time, iteration) # by heng
    ## means3D_offset = pc.get_offset(means3D_rot, time, iteration) # by heng

    # #translation
    # means3D = means3D + means3D_offset # by heng

    # means3D_rot = torch.cat((means3D, rotation_tmp), dim=1)
    # means3D_rot = torch.cat((means3D_ori, rotation_tmp), dim=1)

    # # SE3
    # if not iteration < 3000: # by heng
    #     means3D_homogeneous = rigid.to_homogenous(means3D)  # Convert to homogeneous coordinates
    #     means3D_transformed = torch.bmm(means3D_offset, means3D_homogeneous.unsqueeze(-1))  # Batch matrix multiplication
    #     means3D_transformed = means3D_transformed.squeeze(-1)  # Remove the last dimension to bring back to (batch_size, 4)
    #     means3D = rigid.from_homogenous(means3D_transformed)


    means2D = screenspace_points
    opacity = pc.get_opacity 

    # opacity_mask = pc.get_opacity_mask(means3D, time, iteration) # by heng
    # # opacity_mask = pc.get_opacity_mask(means3D_rot, time, iteration) # by heng
    # opacity = opacity * opacity_mask # by heng

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # scales = pc.get_scaling
        # scales = pc.get_scaling_withOffset(means3D_scaling, time, iteration) # by heng
        scales = pc.scaling_activation(pc._scaling + scale_offset)

        # rotations = pc.get_rotation
        # rotations = pc.get_rotation_ori
        # rotations, rot_offset = pc.get_rotation(means3D_rot, time, iteration) # by heng
        # rotations = pc.get_rotation(means3D, time, iteration) # by heng
        rotations = pc.rotation_activation(pc._rotation + rot_offset)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # shs = pc.get_features

            # shs = mlp_shs.reshape(-1, 16, 3) # by heng
            shs = pc.get_features + mlp_shs.reshape(-1, 16, 3) # by heng

    else:
        colors_precomp = override_color


    # mlp_shs = pc.get_shs(means3D, time, iteration) # by heng
    # shs = shs + mlp_shs # by heng


    # pc.inn.to(means3D.device) # by heng normalizaing flow
    # means3D = pc.inn(means3D)[0] # by heng normalizing flow

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    if save_ply:
        t_id = str(int(viewpoint_camera.time * 1000))
        pc.save_ply_t(os.path.join('test_ply', f'point_cloud_{t_id}.ply'), xyz=means3D, opacities=opacity, rotation=rotations)
        # raise


    # offset_mask = torch.where(torch.norm(means3D_offset, dim=-1) >= 0.1, True, False)
    # print ('number of movement: ', len(means3D_offset[offset_mask]))
    # print (torch.norm(means3D_offset, dim=-1).max(), torch.norm(means3D_offset, dim=-1).min(), viewpoint_camera.time)

    # print ('means3D_offset: ', means3D_offset)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    # offset_mask = torch.where(torch.norm(means3D_offset, dim=-1) < 0.001, True, False) # 0.1 by heng
    # print ('len of means3D_offset: ', len(means3D_offset[offset_mask]))
    # print ('means3D_offset norm: ', torch.norm(means3D_offset, dim=-1).max(), torch.norm(means3D_offset, dim=-1).min())


    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "means3D": means3D,
            "means3D_ori": means3D_ori,
            "rotations": rotations,
            "means3D_offset": means3D_offset,
            "opacities": opacity,
            "rot_offset": rot_offset
            }

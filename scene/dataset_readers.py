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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, time=int(image_name)/len(cam_extrinsics))
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCameras_multiCam(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)


        if int(image_name) > 11:
            continue

        img_mv_path = os.path.dirname(images_folder) + '/mv_images'
        image_mv_name = f'cam{str(int(image_name)+1).zfill(2)}.jpg'

        for f_name in os.listdir(img_mv_path):
            image_path_0 = os.path.join(img_mv_path, f_name, image_mv_name)
            image_0 = Image.open(image_path_0)
            image_name_0 = f'{f_name}_{image_mv_name}'

            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image_0,
                                image_path=image_path_0, image_name=image_name_0, width=width, height=height, time=int(image_name)/len(cam_extrinsics)/2)

        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCameras_nsff(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)

        if int(image_name) > 11:
            continue

        img_mv_path = os.path.dirname(images_folder) + '/mv_images'
        image_mv_name = f'cam{str(int(image_name)+1).zfill(2)}.jpg'

        for f_name in os.listdir(img_mv_path):

            if (int(f_name) % 12) == int(image_name):
                continue

            image_path_0 = os.path.join(img_mv_path, f_name, image_mv_name)

            image_0 = Image.open(image_path_0)
            image_name_0 = f'{f_name}_{image_mv_name}'

            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image_0,
                                image_path=image_path_0, image_name=image_name_0, width=width, height=height, time=int(image_name)/len(cam_extrinsics)/2)


            cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=10): # 8 by heng

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin") 
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin") 
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        # storePly(ply_path, xyz, rgb) 

        # by heng
        num_pts = 100_000
        min_values = np.min(xyz, axis=0)  # Minimum values along the 2nd dimension
        max_values = np.max(xyz, axis=0)
        xyz = np.random.uniform(min_values, max_values, size=(num_pts, 3))
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        # by heng end


    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readColmapSceneInfo_multiCam(path, images, eval, llffhold=8):

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin") 
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin") 
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)


    # print (cam_extrinsics[1])
    # print (cam_extrinsics[13])
    # print (cam_extrinsics[14])
    # print (cam_intrinsics)
    # raise

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras_multiCam(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        # storePly(ply_path, xyz, rgb) 

        # by heng
        num_pts = 100_000
        min_values = np.min(xyz, axis=0)  # Minimum values along the 2nd dimension
        max_values = np.max(xyz, axis=0)
        xyz = np.random.uniform(min_values, max_values, size=(num_pts, 3))
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        # by heng end


    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readColmapSceneInfo_nsff(path, images, eval, llffhold=8):

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin") 
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin") 
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)


    reading_dir = "images" if images == None else images
    cam_infos_unsorted_train = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos_train = sorted(cam_infos_unsorted_train.copy(), key = lambda x : x.image_name)

    cam_infos_unsorted_test = readColmapCameras_nsff(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos_test = sorted(cam_infos_unsorted_test.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos_train)]
        test_cam_infos = [c for idx, c in enumerate(cam_infos_test)]
    else:
        raise
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        # storePly(ply_path, xyz, rgb) 

        # by heng
        num_pts = 100_000
        min_values = np.min(xyz, axis=0)  # Minimum values along the 2nd dimension
        max_values = np.max(xyz, axis=0)
        xyz = np.random.uniform(min_values, max_values, size=(num_pts, 3))
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        # by heng end


    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            #by heng
            cur_time = frame['time'] if 'time' in frame else 1.0 
            #by heng end

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], time=cur_time))




    # # by heng add val 
    # transformsfile_val = 'transforms_val.json'
    # with open(os.path.join(path, transformsfile_val)) as json_file:
    #     contents = json.load(json_file)
    #     fovx = contents["camera_angle_x"]

    #     frames = contents["frames"]
    #     for idx, frame in enumerate(frames):
    #         cam_name = os.path.join(path, frame["file_path"] + extension)

    #         # NeRF 'transform_matrix' is a camera-to-world transform
    #         c2w = np.array(frame["transform_matrix"])
    #         # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    #         c2w[:3, 1:3] *= -1

    #         # get the world-to-camera transform and set R, T
    #         w2c = np.linalg.inv(c2w)
    #         R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    #         T = w2c[:3, 3]

    #         image_path = os.path.join(path, cam_name)
    #         image_name = Path(cam_name).stem
    #         image = Image.open(image_path)

    #         im_data = np.array(image.convert("RGBA"))

    #         bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

    #         norm_data = im_data / 255.0
    #         arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    #         image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

    #         fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
    #         FovY = fovy 
    #         FovX = fovx

    #         #by heng
    #         cur_time = frame['time'] if 'time' in frame else 1.0 
    #         #by heng end

    #         cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
    #                         image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], time=cur_time))

    # # by heng end

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension) #  transforms_test # by heng 

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):

        # # by heng read specific ply
        # plydata = PlyData.read('/home/hengyu/projects/gaussian-splatting_dyn_their/output/standup_dyn/point_cloud/iteration_40000/point_cloud.ply')

        # xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
        #                 np.asarray(plydata.elements[0]["y"]),
        #                 np.asarray(plydata.elements[0]["z"])),  axis=1)

        # num_pts = len(xyz)
        # print(f"Generating random point cloud ({num_pts})...")
        
        # # We create random points inside the bounds of the synthetic Blender scenes
        # shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        # # by heng end


        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfiesCameras(path, setname):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if False: #name.startswith('vrig'): # by heng for val exp
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5

    else:  # for hypernerf
        # train_img = dataset_json['ids'][::4] #4 by heng 
        # if setname == 'train':
        #     train_img = dataset_json['train_ids'] # by heng 
        # else:
        #     train_img = dataset_json['val_ids'] # by heng 

        if setname == 'train':
            train_img = dataset_json['ids'] # by heng 
            # train_img = dataset_json['train_ids'] # by heng 
        else:
            train_img = dataset_json['val_ids'] # by heng 

        all_img = train_img
        ratio = 0.5 #0.5 by heng

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    # all_time = [meta_json[i]['time_id'] for i in all_img]
    all_time = [meta_json[i]['warp_id'] for i in all_img]
    max_time = max(all_time)
    # all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    all_time = [meta_json[i]['warp_id'] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image = np.array(Image.open(image_path))
        image = Image.fromarray((image).astype(np.uint8))
        image_name = Path(image_path).stem

        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              time=fid)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale


def readNerfiesInfo(path, eval):
    print("Reading Nerfies Info")
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path, setname='train')



    if eval:
        # ## for val exp
        # train_cam_infos = cam_infos[:train_num]
        # test_cam_infos = cam_infos[train_num:]

        # train_cam_infos = cam_infos # by heng
        # test_cam_infos, _, _, _ = readNerfiesCameras(path, setname='test') # by heng

        ## for interp exp
        interval = 4 # for interp exp
        all_indices = np.arange(len(cam_infos))
        # train_indices = all_indices[::interval]
        # test_indices = (train_indices[:-1] + train_indices[1:]) // 2

        # Calculate the number of splits
        num_splits = len(all_indices) // interval
        train_indices = []
        test_indices = []
        for i in range(num_splits):
            start = i * interval
            end = start + interval - 1  # Take the first 3 indices as train
            train_indices.extend(all_indices[start:end])
            test_indices.append(all_indices[end])

        train_cam_infos = [cam_infos[i] for i in train_indices]
        test_cam_infos = [cam_infos[i] for i in test_indices]

        train_cam_infos = cam_infos # only for render demo using all ids

        # print (len(cam_infos))
        # print (len(train_cam_infos), len(test_cam_infos))
        # print (train_num)
        # raise
        # train_cam_time = [i.time for i in train_cam_infos]
        # test_cam_time = [i.time for i in test_cam_infos]
        # print (train_cam_time)
        # print (test_cam_time)
        # raise

    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # print (len(train_cam_infos))
    # print (len(test_cam_infos))
    # raise

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")


    if not os.path.exists(ply_path):
        print(f"Generating point cloud from nerfies...")

        npy_path = os.path.join(path, "points.npy")
        if not os.path.exists(npy_path):
            bin_path = os.path.join(path, "colmap/sparse/0/points3D.bin")

            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                raise
        else:    
            xyz = np.load(npy_path)
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0


        # by heng
        num_pts_addi = 100_000
        min_values = np.min(xyz, axis=0)  # Minimum values along the 2nd dimension
        max_values = np.max(xyz, axis=0)
        xyz_addi = np.random.uniform(min_values, max_values, size=(num_pts_addi, 3))
        shs_addi = np.random.random((num_pts_addi, 3)) / 255.0

        xyz = np.concatenate([xyz, xyz_addi], axis=0)
        shs = np.concatenate([shs, shs_addi], axis=0)
        # by heng end


        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info




# sceneLoadTypeCallbacks = {
#     "Colmap": readColmapSceneInfo, #readColmapSceneInfo, # by heng # readColmapSceneInfo_multiCam # readColmapSceneInfo_nsff
#     "Blender" : readNerfSyntheticInfo
# }

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    # "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    # "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
}

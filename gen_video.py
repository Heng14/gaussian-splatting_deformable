import os
import numpy as np
import cv2
import imageio
import skvideo.io


# img_path = '/home/hengyu/projects/gaussian-splatting_dyn_co_raster_mask_offsetAsCtrlSig/output/bouncingball_pureT_OffReg_hardpool_sortL1Loss_usePureMask_r024/train/ours_51000/renders_pureT_r011'
img_path = 'output/aleks-teapot_offsetnormReg0.01_randInit_shsOff_2x_200k/train/ours_120000/renders'


# fname = f'attribute_{start}_{end}_{sample_rate}'

fps = '30'
# fps = '15'
img_list = os.listdir(img_path)
img_list.sort()

frames_out = []
for frame_id, frame in enumerate(img_list):
    # frame_name = str(frame_id).zfill(5)
    # if frame_id < 110 or frame_id > 200:
    #     continue
    # print (frame)
    im1 = imageio.imread(f'{img_path}/{frame}')
    frames_out.append(im1)    

skvideo.io.vwrite(
f"{img_path}/out.mp4",
frames_out,
inputdict={"-r": fps},
outputdict={"-r": fps},
)  



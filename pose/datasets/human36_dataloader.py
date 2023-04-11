
'''
    file:   hum36m_dataloader.py

    author: zhangxiong(1025679612@qq.com)
    date:   2018_05_09
    purpose:  load hum3.6m data
'''
from __future__ import print_function, absolute_import

import sys
from torch.utils.data import Dataset, DataLoader
import os 
import glob
import numpy as np
import random
import cv2
import json
import h5py
import torch

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

# sys.path.append('./src')
# from utils import calc_aabb,center_crop,hog_box, cut_image, cut_image_2, flip_image, draw_lsp_14kp__bone, rectangle_intersect, get_rectangle_intersect_ratio, convert_image_by_pixformat_normalize, reflect_pose, reflect_lsp_kp
from .utils import calc_aabb,center_crop,hog_box, cut_image, cut_image_2, flip_image, draw_lsp_14kp__bone, rectangle_intersect, get_rectangle_intersect_ratio, convert_image_by_pixformat_normalize, reflect_pose, reflect_lsp_kp
# from config import args
# from timer import Clock
from .data_config import args
from .timer import Clock
# for resnet 224,224 image and heat 56
# for stacked 256,256 image ansd heat 64
N_KEYPOINTS = 16
N_IMG_CHANNELS = 3
RAW_IMG_SIZE = 1000
# MODEL_IMG_SIZE = 56 #56 for resnwt
MODEL_IMG_SIZE = 64 #56 for reset_with 5 deconv layers
DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1966, 0.1734, 0.1836]
MODEL_NEURONS = 16

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def draw_labelmap(img, pt, sigma=1, type='Gaussian'): # this functio is used in mpii dataset but not
    # working for human36m dataset why ????????? throws error in some cases
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img), 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    """img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
ValueError: could not broadcast input array from shape (7,7) into shape (8,7)"""
# this error after some time of training
    return to_torch(img), 1


def vector_to_heatmaps(keypoints):
    """
    Creates 2D heatmaps from keypoint locations for a single image
    Input: array of size N_KEYPOINTS x 2
    Output: array of size N_KEYPOINTS x MODEL_IMG_SIZE x MODEL_IMG_SIZE
    """
    heatmaps = np.zeros([N_KEYPOINTS, MODEL_IMG_SIZE, MODEL_IMG_SIZE])
    for k, (x, y) in enumerate(keypoints):
        x, y = int(x * MODEL_IMG_SIZE), int(y * MODEL_IMG_SIZE)
        if (0 <= x < MODEL_IMG_SIZE) and (0 <= y < MODEL_IMG_SIZE):
            heatmaps[k, int(y), int(x)] = 1

    heatmaps = blur_heatmaps(heatmaps) #to check no smoothing on loss
    return heatmaps


def blur_heatmaps(heatmaps):
    """Blurs heatmaps using GaussinaBlur of defined size"""
    heatmaps_blurred = heatmaps.copy()
    for k in range(len(heatmaps)):
        if heatmaps_blurred[k].max() == 1:
            # heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], (51, 51), 1)
            heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k],(0,0), 2)
            heatmaps_blurred[k] = heatmaps_blurred[k] / heatmaps_blurred[k].max()
    return heatmaps_blurred

# def gaussian(xL, yL, H, W, sigma=1):

#     channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
#     channel = np.array(channel, dtype=np.float32)
#     channel = np.reshape(channel, newshape=(H, W))

#     return channel



def human36m_to_common(kps):
    # torch_arr = kps.numpy()
    torch_arr = kps.copy()
    #tensor to nump
    updated_tensor = np.delete(torch_arr, [7], axis=0)
    return torch.from_numpy(updated_tensor)


class hum36m_dataloader(Dataset):
    def __init__(self, data_set_path,annotaion_path, use_crop, scale_range, use_flip, min_pts_required, pix_format = 'NHWC', normalize = True, flip_prob = 0.3):
        self.data_folder = data_set_path
        self.anno_file_path = annotaion_path
        self.use_crop = use_crop
        self.scale_range = scale_range
        self.use_flip = use_flip
        self.flip_prob = flip_prob
        self.min_pts_required = min_pts_required
        self.pix_format = pix_format
        # self.normalize = normalize
        self.normalize = True # usually false
        self._load_data_set()

    def _load_data_set(self):
        
        # clk = Clock()

        self.images = []
        self.kp2ds  = []
        self.boxs   = []
        self.kp3ds  = []
        self.shapes = []
        self.poses  = []
        self.leftTop= []

        print('start loading hum3.6m data.')

        anno_file_path = os.path.join(self.data_folder,self.anno_file_path)
        # anno_file_path = os.path.join(self.data_folder, 'cameras.h5')
        # with h5py.File(anno_file_path) as fp:
        with np.load(anno_file_path, allow_pickle=True) as fp:
            # import pdb;pdb.set_trace()
            total_center = np.array(fp['center']) #total_pose
            total_kp2d = np.array(fp['part'])
            total_kp3d = np.array(fp['S'])
            total_scale = np.array(fp['scale']) #total_shape
            total_image_names = np.array(fp['imgname'])

            # import pdb;pdb.set_trace()
            assert len(total_kp2d) == len(total_kp3d) and len(total_kp2d) == len(total_image_names) and \
                len(total_kp2d) == len(total_scale) and len(total_kp2d) == len(total_center)

            # l =  110232
            l = len(total_kp2d)
            def _collect_valid_pts(pts):
                r = []
                for pt in pts:
                    # if pt[2] != 0:
                    if pt[1] != 0:
                        r.append(pt)
                return r

            for index in range(l):
                # kp2d = total_kp2d[index].reshape((-1, 3))
                kp2d = total_kp2d[index].reshape((-1, 2))
                # import pdb;pdb.set_trace()
                # if np.sum(kp2d[:, 2]) < self.min_pts_required: # chnaged to make the code work
                    # continue
                if np.sum(kp2d[:, 1]) < self.min_pts_required:
                    continue
                # lt, rb = hog_box(os.path.join(self.data_folder, 'images/') + total_image_names[index])

                lt, rb, v = calc_aabb(_collect_valid_pts(kp2d))
                self.kp2ds.append(np.array(kp2d.copy(), dtype = np.float))
                self.boxs.append((lt, rb))
                # import pdb;pdb.set_trace()
                # self.kp3ds.append(total_kp3d[index].copy().reshape(-1, 3))
                self.kp3ds.append(total_kp3d[index].copy().reshape(-1, 3))
                self.shapes.append(total_scale[index].copy())
                self.poses.append(total_center[index].copy())
                # self.images.append(os.path.join(self.data_folder, 'image') + total_image_names[index].decode())
                self.images.append(os.path.join(self.data_folder, 'images/') + total_image_names[index])

        print('finished load hum3.6m data, total {} samples'.format(len(self.kp3ds)))
        
        # clk.stop()

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        image_path = self.images[index]
        kps = self.kp2ds[index].copy()
        kps_original = self.kp2ds[index].copy()
        box = self.boxs[index]
        kp_3d = self.kp3ds[index].copy()
        center = self.poses[index]

        # import pdb;pdb.set_trace()
        scale = np.random.rand(4) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        # scale = [1,1,1,1]
        image, kps, leftTop = cut_image(image_path, kps, scale, box[0], box[1])
        # cv2.imwrite(filename + 'crop_box.jpg', image)
        # import pdb;pdb.set_trace()
        # image, kps, center = center_crop(image_path, kps, center)
        # image = cv2.imread(image_path)

        # kps_unnorm = unoff_set_pts(kps,leftTop)
        ratio_x = 1.0 * 1/image.shape[0] # image.shape[0]
        ratio_y = 1.0 * 1/image.shape[1] # image.shape[1]
        # kps[:, :1] *= ratio_x 
        # kps[:, 1:2] *= ratio_y
        kps[:, :1] *= ratio_x  
        kps[:, 1:2] *= ratio_y 
        dst_image = cv2.resize(image, (256,256), interpolation = cv2.INTER_CUBIC) # for stacked 256,256
        kps = human36m_to_common(kps)
        kps_original = human36m_to_common(kps_original)
        heatmaps = vector_to_heatmaps(kps)
        # target = torch.zeros(17, 64, 64)
        # for i in range(17):
        #     target[i],vis = draw_labelmap( target[i],kps[i] -1, sigma=1, type='Gaussian')

        heatmaps = torch.from_numpy(np.float32(heatmaps))
        trival, shape, pose = np.zeros(3), self.shapes[index], self.poses[index]

        if self.use_flip and random.random() <= self.flip_prob:
            dst_image, kps = flip_image(dst_image, kps)
            pose = reflect_pose(pose)
            kp_3d = reflect_lsp_kp(kp_3d)

        #normalize kp to [-1, 1]
        # ratio = 1.0 / args.crop_size
        # kps[:, :2] = 2.0 * kps[:, :2] * ratio - 1.0
        # kps1 = 2.0 * kps[:, :2] * ratio - 1.0
        # kps2 = kps[:, :2] / args.crop_size * 2 - [1, 1]
        # normalize kp_3d to [-1, 1]
        # min_values =  np.min(kp_3d[:, :3], axis=0)
        # max_values =  np.max(kp_3d[:, :3], axis=0)

        # kp_3d[:, :3] = (kp_3d[:, :3] - min_values) * 2 / (max_values - min_values) - 1
        kp_3d[1:] -= kp_3d[:1]
        # import pdb; pdb.set_trace()
        kp_3d_u = human36m_to_common(kp_3d) # for 16 keypoints to match with mpii dataset
        # pos_3d[:, 1:] -= pos_3d[:, :1] 
        # theta = np.concatenate((trival, pose, shape), axis = 0)

        return {
            'image': torch.from_numpy(convert_image_by_pixformat_normalize(dst_image, self.pix_format, self.normalize)).float(),
            # 'kp_2d': torch.from_numpy(kps).float(),
            'kp_2d': (kps).float(),
            'kp_2d_org': (kps_original).float(),
            'kp_3d': torch.from_numpy(kp_3d).float(),
            'kp_3d_u': (kp_3d_u).float(), # for 16 keypoints
            # 'theta': torch.from_numpy(theta).float(),
            'image_name': self.images[index],
            'w_smpl':1.0,
            'w_3d':1.0,
            'heatmap': heatmaps,
            # 'heatmap': target,
            'leftTop':leftTop,
            'ratio_x': ratio_x,
            'ratio_y': ratio_y,
            'data_set':'hum3.6m'
        }

def hm36(**kwargs):
    return hum36m_dataloader(**kwargs)

hm36.njoints = 17

if __name__ == '__main__':
    # h36m = hum36m_dataloader('/netscratch/nafis/human-pose/human36_dataset', True, [1.1, 2.0], True, 5, flip_prob = 1)
    print("start")
    annotation_path_train = "annotation_body3d/fps50/h36m_train.npz"
    annotation_path_test = "annotation_body3d/fps50/h36m_test.npz"
    root_data_path = '/netscratch/nafis/human-pose/dataset_hum36m_f50'
    h36m = hum36m_dataloader(root_data_path, annotation_path_train,True, [1.1, 2.0], False, 5, flip_prob = 1)
    l = len(h36m)
    filename = '/netscratch/nafis/human-pose/real_time_pose/results/delete'
    for _ in range(l):
        r = h36m.__getitem__(_)
        # import pdb;pdb.set_trace()
        print(r['image'].permute(2,0,1).shape)
        # import pdb;pdb.set_trace()
        cvimg = cv2.imread(r['image_name'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        print(f'cvimage {cvimg.shape}')
        kps2d = r['kp_2d']
        heatmap = r['heatmap']
        print(heatmap.shape)
        plt.plot(heatmap.sum(axis=0))
        plt.savefig(filename + '_heat.jpg')
        # import pdb;pdb.set_trace()
        # lt, rb, v = calc_aabb(collect_valid_pts(kps2d))
        # scale = np.random.rand(4) * (2.0 - 1.1) + 1.1
        # image, kps = cut_image(r['image_name'], kps2d, scale, lt, rb)
        print(r['image_name'])
        
    
        kps2d = r['kp_2d']
        # import pdb;pdb.set_trace()
        
        # generate heat map
        # heatmaps = []
        # for i in range(0,17):
        #     x = kps2d[i][0]
        #     y = kps2d[i][1]
        #     heatmap = gaussian(x, y, 54, 54)
        #     heatmaps.apped(heatmap)

        tmpkps = r['kp_3d']
        print("3d points coordinate")

        # for i in tmpkps:
        #     print(i)
        # tmpkps[:,0], tmpkps[:,1] = r['kp_3d'][:,0], r['kp_3d'][:,1]
        # tmpkps[:,2] = 1
        # print(tmpkps.shape) # (17,3)
        skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
        #convert key points to pixel coordinate before visualize
        R = np.array([[-0.91536173,  0.05154812, -0.39931903],
                    [ 0.40180837,  0.18037357, -0.89778361],
                    [ 0.02574754, -0.98224649, -0.18581953]])
        f = np.array([[1145.04940459],[1143.78109572]]) #for camera 54138969
        c = np.array([[512.54150496],[515.45148698]]) #for camera 54138969
        # f = np.array([[1145.51133842],[1144.77392808]]) #for camera  60457274
        # c = np.array([[514.96819732], [501.88201854]])  #for camera  60457274
        T = np.array([[1.84110703],
                    [4.95528462],
                    [1.5634454 ]])
        k = np.array([[-0.20709891],
                    [ 0.24777518],
                    [-0.00307515]])
        print("#########")
        # d3kps = world2cam(tmpkps, R, T)
        # print(tmpkps.sha)
        # vis_3d_skeleton(tmpkps, skeleton)
        tmpkps[1:] += tmpkps[:1]

        # tmpkps = cam2pixel(tmpkps, f, c) # only in cam coordinate
        # print(f'kps_3d after pixel.shape {tmpkps.shape}')  #tmpkps shape  (17,3)
        # tmpimg = vis_keypoints(cvimg, tmpkps.transpose(1,0), skeleton)
        # cv2.imwrite(filename + 'box.jpg', tmpimg)
        # # cv2.imwrite(filename + 'box.jpg', cvimg)

        # for i in kps_2d:
        #     print(i)
        #     # cvimg = cv2.circle(cvimg, (i[0],i[1]), radius=3, color=(0,0,255), thickness=-1)
        #     # cv2.imwrite(filename + '2d_trail.jpg', cvimg)
        # print("3d points on pix coordinate")
        # for i in tmpkps:
        #     print(i)
        # kps_2d = pixel2cam(kps_2d, f, c)
        # for i in kps_2d:
        #     print(i)
        break
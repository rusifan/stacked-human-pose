
'''
    file:   util.py

    date:   2018_04_29
    author: zhangxiong(1025679612@qq.com)
'''

import h5py
import torch
import numpy as np
# from .config import args for dataloader when training
from .config import args
import json
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import math
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import io
from PIL import Image

N_KEYPOINTS = 17
N_IMG_CHANNELS = 3
RAW_IMG_SIZE = 224
MODEL_IMG_SIZE = 56
DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1966, 0.1734, 0.1836]
MODEL_NEURONS = 16

def load_mean_theta():
    mean = np.zeros(args.total_theta_count, dtype = np.float)

    mean_values = h5py.File(args.smpl_mean_theta_path)
    mean_pose = mean_values['pose']
    mean_pose[:3] = 0
    mean_shape = mean_values['shape']
    mean_pose[0]=np.pi

    #init sacle is 0.9
    mean[0] = 0.9

    mean[3:75] = mean_pose[:]
    mean[75:] = mean_shape[:]

    return mean

def batch_rodrigues(theta):
    #theta N x 3
    batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    
    return quat2mat(quat)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def batch_global_rigid_transformation(Rs, Js, parent, rotate_base = False):
    N = Rs.shape[0]
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype = np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = Variable(torch.from_numpy(np_rot_x).float()).cuda()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).cuda()], dim = 1)
        return torch.cat([R_homo, t_homo], 2)
    
    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim = 1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 24, 1, 1)).cuda()], dim = 2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A


def batch_lrotmin(theta):
    theta = theta[:,3:].contiguous()
    Rs = batch_rodrigues(theta.view(-1, 3))
    print(Rs.shape)
    e = Variable(torch.eye(3).float())
    Rs = Rs.sub(1.0, e)

    return Rs.view(-1, 23 * 9)

def batch_orth_proj(X, camera):
    '''
        X is N x num_points x 3
    '''
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    return (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)

def calc_aabb(ptSets):
    if not ptSets or len(ptSets) == 0:
        return False, False, False
    
    ptLeftTop     = np.array([ptSets[0][0], ptSets[0][1]])
    ptRightBottom = ptLeftTop.copy()
    for pt in ptSets:
        ptLeftTop[0]     = min(ptLeftTop[0], pt[0])
        ptLeftTop[1]     = min(ptLeftTop[1], pt[1])
        ptRightBottom[0] = max(ptRightBottom[0], pt[0])
        ptRightBottom[1] = max(ptRightBottom[1], pt[1])

    return ptLeftTop, ptRightBottom, len(ptSets) >= 5

'''
    calculate a obb for a set of points

    inputs: 
        ptSets: a set of points

    return the center and 4 corners of a obb
'''
def calc_obb(ptSets):
    ca = np.cov(ptSets,y = None,rowvar = 0,bias = 1)
    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)
    ar = np.dot(ptSets,np.linalg.inv(tvect))
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)
    diff    = (maxa - mina)*0.5
    center  = mina + diff
    corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]]])
    corners = np.dot(corners, tvect)
    return corners[0], corners[1], corners[2], corners[3]

def get_image_cut_box(leftTop, rightBottom, ExpandsRatio, Center = None):
    try:
        l = len(ExpandsRatio)
    except:
        ExpandsRatio = [ExpandsRatio, ExpandsRatio, ExpandsRatio, ExpandsRatio]

    def _expand_crop_box(lt, rb, scale):
        center = (lt + rb) / 2.0
        xl, xr, yt, yb = lt[0] - center[0], rb[0] - center[0], lt[1] - center[1], rb[1] - center[1]
        xl, xr, yt, yb = xl * scale[0], xr * scale[1], yt * scale[2], yb * scale[3]
        #expand it
        lt, rb = np.array([center[0] + xl, center[1] + yt]), np.array([center[0] + xr, center[1] + yb])
        lb, rt = np.array([center[0] + xl, center[1] + yb]), np.array([center[0] + xr, center[1] + yt])
        center = (lt + rb) / 2
        return center, lt, rt, rb, lb

    if Center == None:
        Center = (leftTop + rightBottom) // 2

    Center, leftTop, rightTop, rightBottom, leftBottom = _expand_crop_box(leftTop, rightBottom, ExpandsRatio)

    offset = (rightBottom - leftTop) // 2

    cx = offset[0]
    cy = offset[1]

    r = max(cx, cy)

    cx = r
    cy = r
    
    x = int(Center[0])
    y = int(Center[1])

    return [x - cx, y - cy], [x + cx, y + cy]

def shrink(leftTop, rightBottom, width, height):
    xl = -leftTop[0]
    xr = rightBottom[0] - width

    yt = -leftTop[1]
    yb = rightBottom[1] - height

    cx = (leftTop[0] + rightBottom[0]) / 2
    cy = (leftTop[1] + rightBottom[1]) / 2

    r = (rightBottom[0] - leftTop[0]) / 2

    sx = max(xl, 0) + max(xr, 0)
    sy = max(yt, 0) + max(yb, 0)

    if (xl <= 0 and xr <= 0) or (yt <= 0 and yb <=0):
        return leftTop, rightBottom
    elif leftTop[0] >= 0 and leftTop[1] >= 0 : # left top corner is in box
        l = min(yb, xr)
        r = r - l / 2
        cx = cx - l / 2
        cy = cy - l / 2
    elif rightBottom[0] <= width and rightBottom[1] <= height : # right bottom corner is in box
        l = min(yt, xl)
        r = r - l / 2
        cx = cx + l / 2
        cy = cy + l / 2
    elif leftTop[0] >= 0 and rightBottom[1] <= height : #left bottom corner is in box
        l = min(xr, yt)
        r = r - l  / 2
        cx = cx - l / 2
        cy = cy + l / 2
    elif rightBottom[0] <= width and leftTop[1] >= 0 : #right top corner is in box
        l = min(xl, yb)
        r = r - l / 2
        cx = cx + l / 2
        cy = cy - l / 2
    elif xl < 0 or xr < 0 or yb < 0 or yt < 0:
        return leftTop, rightBottom
    elif sx >= sy:
        sx = max(xl, 0) + max(0, xr)
        sy = max(yt, 0) + max(0, yb)
        # cy = height / 2
        if yt >= 0 and yb >= 0:
            cy = height / 2
        elif yt >= 0:
            cy = cy + sy / 2
        else:
            cy = cy - sy / 2
        r = r - sy / 2
        
        if xl >= sy / 2 and xr >= sy / 2:
            pass
        elif xl < sy / 2:
            cx = cx - (sy / 2 - xl)
        else:
            cx = cx + (sy / 2 - xr)
    elif sx < sy:
        cx = width / 2
        r = r - sx / 2
        if yt >= sx / 2 and yb >= sx / 2:
            pass
        elif yt < sx / 2:
            cy = cy - (sx / 2 - yt)
        else:
            cy = cy + (sx / 2 - yb)
        

    return [cx - r, cy - r], [cx + r, cy + r]

'''
    offset the keypoint by leftTop
'''
def off_set_pts(keyPoints, leftTop):
    result = keyPoints.copy()
    result[:, 0] -= leftTop[0]
    result[:, 1] -= leftTop[1]
    return result

'''
    cut the image, by expanding a bounding box
'''
def hog_box(image):
    # image = cv2.imread(image_path)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # import pdb;pdb.set_trace()
    boxes, weights = hog.detectMultiScale(image, winStride=(8,8) )
    boxes = np.array([[x - 80, y, x + w + 80, y + h] for (x, y, w, h) in boxes])
    # import pdb;pdb.set_trace()
    # print(boxes.shape)
    if np.any(boxes):
        return [boxes[0][0], boxes[0][1]], [boxes[0][2], boxes[0][3]]
    else :
        return [0, 0] , [image.shape[0], image.shape[1]]
    # return [boxes[0], boxes[1]], [boxes[2], boxes[3]]

def center_crop(filePath, kps, center):
    originImage = cv2.imread(filePath)
    cx, cy = int(center[0]), int(center[1])
    x, y, w, h = cx - 200, cy - 200, 400, 400
    cropped_img = originImage[y:y+h, x:x+w]

    return cropped_img, off_set_pts(kps, [x,y]), [x,y]    
    

def cut_image(filePath, kps, expand_ratio, leftTop, rightBottom, hf):
    # originImage = cv2.imread(filePath)
    # import pdb;pdb.set_trace()
    bin_img = np.array(hf[filePath])
    originImage = np.array(Image.open(io.BytesIO(bin_img)))[..., :3]

    height       = originImage.shape[0]
    width        = originImage.shape[1]
    channels     = originImage.shape[2] if len(originImage.shape) >= 3 else 1
    # leftTop, rightBottom = hog_box(originImage)
    leftTop, rightBottom = get_image_cut_box(leftTop, rightBottom, expand_ratio)
    
    #remove extra space.
    #leftTop, rightBottom = shrink(leftTop, rightBottom, width, height)

    lt = [int(leftTop[0]), int(leftTop[1])]
    rb = [int(rightBottom[0]), int(rightBottom[1])]

    lt[0] = max(0, lt[0])
    lt[1] = max(0, lt[1])
    rb[0] = min(rb[0], width)
    rb[1] = min(rb[1], height)

    leftTop      = [int(leftTop[0]), int(leftTop[1])]
    rightBottom  = [int(rightBottom[0] + 0.5), int(rightBottom[1] + 0.5)]

    dstImage = np.zeros(shape = [rightBottom[1] - leftTop[1], rightBottom[0] - leftTop[0], channels], dtype = np.uint8)
    dstImage[:,:,:] = 0

    offset = [lt[0] - leftTop[0], lt[1] - leftTop[1]]
    size   = [rb[0] - lt[0], rb[1] - lt[1]]

    dstImage[offset[1]:size[1] + offset[1], offset[0]:size[0] + offset[0], :] = originImage[lt[1]:rb[1], lt[0]:rb[0],:]
    return dstImage, off_set_pts(kps, leftTop), leftTop

def cut_image_2(filePath, kps, expand_ratio, leftTop, rightBottom):
    originImage = cv2.imread(filePath)
    # import pdb;pdb.set_trace()
    height       = originImage.shape[0]
    width        = originImage.shape[1]
    channels     = originImage.shape[2] if len(originImage.shape) >= 3 else 1
    # leftTop, rightBottom = hog_box(originImage)
    # leftTop, rightBottom = get_image_cut_box(leftTop, rightBottom, expand_ratio)
    
    #remove extra space.
    #leftTop, rightBottom = shrink(leftTop, rightBottom, width, height)

    lt = [int(leftTop[0]), int(leftTop[1])]
    rb = [int(rightBottom[0]), int(rightBottom[1])]

    lt[0] = max(0, lt[0])
    lt[1] = max(0, lt[1])
    rb[0] = min(rb[0], width)
    rb[1] = min(rb[1], height)

    leftTop      = [int(leftTop[0]), int(leftTop[1])]
    rightBottom  = [int(rightBottom[0] + 0.5), int(rightBottom[1] + 0.5)]

    dstImage = np.zeros(shape = [rightBottom[1] - leftTop[1], rightBottom[0] - leftTop[0], channels], dtype = np.uint8)
    dstImage[:,:,:] = 0

    offset = [lt[0] - leftTop[0], lt[1] - leftTop[1]]
    size   = [rb[0] - lt[0], rb[1] - lt[1]]

    dstImage[offset[1]:size[1] + offset[1], offset[0]:size[0] + offset[0], :] = originImage[lt[1]:rb[1], lt[0]:rb[0],:]
    return dstImage, off_set_pts(kps, leftTop)
'''
    purpose:
        reflect key point, when the image is reflect by left-right
    
    inputs:
        kps:3d key point(14 x 3)

    marks:
        the key point is given by lsp order.
'''
def reflect_lsp_kp(kps):
    kp_map = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
    joint_ref = kps[kp_map]
    joint_ref[:,0] = -joint_ref[:,0]

    return joint_ref - np.mean(joint_ref, axis = 0)

'''
    purpose:
        reflect poses, when the image is reflect by left-right
    
    inputs:
        poses: 72 real number
'''
def reflect_pose(poses):
    swap_inds = np.array([
            0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18,
            19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32,
            36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49,
            50, 57, 58, 59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66,
            67, 68
    ])

    sign_flip = np.array([
            1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
            -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
            -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
            1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
            -1, 1, -1, -1
    ])

    return poses[swap_inds] * sign_flip

'''
    purpose:
        crop the image
    inputs:
        image_path: the 
'''
def crop_image(image_path, angle, lt, rb, scale, kp_2d, crop_size):
    '''
        given a crop box, expand it at 4 directions.(left, right, top, bottom)
    '''
    assert 'error algorithm exist.' and 0

    def _expand_crop_box(lt, rb, scale):
        center = (lt + rb) / 2.0
        xl, xr, yt, yb = lt[0] - center[0], rb[0] - center[0], lt[1] - center[1], rb[1] - center[1]
        xl, xr, yt, yb = xl * scale[0], xr * scale[1], yt * scale[2], yb * scale[3]
        #expand it
        lt, rb = np.array([center[0] + xl, center[1] + yt]), np.array([center[0] + xr, center[1] + yb])
        lb, rt = np.array([center[0] + xl, center[1] + yb]), np.array([center[0] + xr, center[1] + yt])
        center = (lt + rb) / 2
        return center, lt, rt, rb, lb
    
    '''
        extend the box to square
    '''
    def _extend_box(center, lt, rt, rb, lb, crop_size):
        lx, ly = np.linalg.norm(rt - lt), np.linalg.norm(lb - lt)
        dx, dy = (rt - lt) / lx, (lb - lt) / ly
        l = max(lx, ly) / 2.0
        return center - l * dx - l * dy, center + l * dx - l *dy, center + l * dx + l * dy, center - l * dx + l * dy, dx, dy, crop_size * 1.0 / l

    def _get_sample_points(lt, rt, rb, lb, crop_size):
        vec_x = rt - lt
        vec_y = lb - lt
        i_x, i_y = np.meshgrid(range(crop_size), range(crop_size))
        i_x = i_x.astype(np.float)
        i_y = i_y.astype(np.float)
        i_x /= float(crop_size)
        i_y /= float(crop_size)
        interp_points = i_x[..., np.newaxis].repeat(2, axis=2) * vec_x + i_y[..., np.newaxis].repeat(2, axis=2) * vec_y
        interp_points += lt
        return interp_points

    def _sample_image(src_image, interp_points):
        sample_method = 'nearest'
        interp_image = np.zeros((interp_points.shape[0] * interp_points.shape[1], src_image.shape[2]))
        i_x = range(src_image.shape[1])
        i_y = range(src_image.shape[0])
        flatten_interp_points = interp_points.reshape([interp_points.shape[0]*interp_points.shape[1], 2])
        for i_channel in range(src_image.shape[2]):
            interp_image[:, i_channel] = interpolate.interpn((i_y, i_x), src_image[:, :, i_channel],
                                                            flatten_interp_points[:, [1, 0]], method = sample_method,
                                                            bounds_error=False, fill_value=0)
        interp_image = interp_image.reshape((interp_points.shape[0], interp_points.shape[1], src_image.shape[2]))
    
        return interp_image
    
    def _trans_kp_2d(kps, center, dx, dy, lt, ratio):
        kp2d_offset = kps[:, :2] - center
        proj_x, proj_y = np.dot(kp2d_offset, dx), np.dot(kp2d_offset, dy)
        #kp2d = (dx * proj_x + dy * proj_y + lt) * ratio
        for idx in range(len(kps)):
            kps[idx, :2] = (dx * proj_x[idx] + dy * proj_y[idx] + lt) * ratio
        return kps


    src_image = cv2.imread(image_path)

    center, lt, rt, rb, lb  = _expand_crop_box(lt, rb, scale)

    #calc rotated box
    radian = angle * np.pi / 180.0
    v_sin, v_cos = math.sin(radian), math.cos(radian)

    rot_matrix = np.array(
        [
            [v_cos, v_sin],
            [-v_sin, v_cos]
        ]
    )

    n_corner = (np.dot(rot_matrix, np.array([lt - center, rt - center, rb - center, lb - center]).T).T) + center
    n_lt, n_rt, n_rb, n_lb = n_corner[0], n_corner[1], n_corner[2], n_corner[3] 
    
    lt, rt, rb, lb = calc_obb(np.array([lt, rt, rb, lb, n_lt, n_rt, n_rb, n_lb]))
    lt, rt, rb, lb, dx, dy, ratio = _extend_box(center, lt, rt, rb, lb, crop_size = crop_size)
    s_pts = _get_sample_points(lt, rt, rb, lb, crop_size)
    dst_image = _sample_image(src_image, s_pts)
    kp_2d = _trans_kp_2d(kp_2d, center, dx, dy, lt, ratio)

    return dst_image, kp_2d


'''
    purpose:
        flip a image given by src_image and the 2d keypoints
    flip_mode: 
        0: horizontal flip
        >0: vertical flip
        <0: horizontal & vertical flip
'''
def flip_image(src_image, kps):
    h, w = src_image.shape[0], src_image.shape[1]
    src_image = cv2.flip(src_image, 1)
    if kps is not None:
        kps[:, 0] = w - 1 - kps[:, 0]
        kp_map = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
        kps[:, :] = kps[kp_map]

    return src_image, kps

'''
    src_image: h x w x c
    pts: 14 x 3
'''
def draw_lsp_14kp__bone(src_image, pts):
        bones = [
            [0, 1, 255, 0, 0],
            [1, 2, 255, 0, 0],
            [2, 12, 255, 0, 0],
            [3, 12, 0, 0, 255],
            [3, 4, 0, 0, 255],
            [4, 5, 0, 0, 255],
            [12, 9, 0, 0, 255],
            [9,10, 0, 0, 255],
            [10,11, 0, 0, 255],
            [12, 8, 255, 0, 0],
            [8,7, 255, 0, 0],
            [7,6, 255, 0, 0],
            [12, 13, 0, 255, 0]
        ]

        for pt in pts:
            if pt[2] > 0.2:
                cv2.circle(src_image,(int(pt[0]), int(pt[1])),2,(0,255,255),-1)
        
        for line in bones:
            pa = pts[line[0]]
            pb = pts[line[1]]
            xa,ya,xb,yb = int(pa[0]),int(pa[1]),int(pb[0]),int(pb[1])
            if pa[2] > 0.2 and pb[2] > 0.2:
                cv2.line(src_image,(xa,ya),(xb,yb),(line[2], line[3], line[4]),2)  

'''
    return whether two segment intersect
'''

def line_intersect(sa, sb):
    al, ar, bl, br = sa[0], sa[1], sb[0], sb[1]
    assert al <= ar and bl <= br
    if al >= br or bl >= ar:
        return False
    return True

'''
    return whether two rectangle intersect
    ra, rb left_top point, right_bottom point
'''
def rectangle_intersect(ra, rb):
    ax = [ra[0][0], ra[1][0]]
    ay = [ra[0][1], ra[1][1]]

    bx = [rb[0][0], rb[1][0]]
    by = [rb[0][1], rb[1][1]]

    return line_intersect(ax, bx) and line_intersect(ay, by)

def get_intersected_rectangle(lt0, rb0, lt1, rb1):
    if not rectangle_intersect([lt0, rb0], [lt1, rb1]):
        return None, None

    lt = lt0.copy()
    rb = rb0.copy()

    lt[0] = max(lt[0], lt1[0])
    lt[1] = max(lt[1], lt1[1])

    rb[0] = min(rb[0], rb1[0])
    rb[1] = min(rb[1], rb1[1])
    return lt, rb

def get_union_rectangle(lt0, rb0, lt1, rb1):
    lt = lt0.copy()
    rb = rb0.copy()

    lt[0] = min(lt[0], lt1[0])
    lt[1] = min(lt[1], lt1[1])

    rb[0] = max(rb[0], rb1[0])
    rb[1] = max(rb[1], rb1[1])
    return lt, rb

def get_rectangle_area(lt, rb):
    return (rb[0] - lt[0]) * (rb[1] - lt[1])

def get_rectangle_intersect_ratio(lt0, rb0, lt1, rb1):
    (lt0, rb0), (lt1, rb1) = get_intersected_rectangle(lt0, rb0, lt1, rb1), get_union_rectangle(lt0, rb0, lt1, rb1)

    if lt0 is None:
        return 0.0
    else:
        return 1.0 * get_rectangle_area(lt0, rb0) / get_rectangle_area(lt1, rb1)

def convert_image_by_pixformat_normalize(src_image, pix_format, normalize):
    if pix_format == 'NCHW':
        src_image = src_image.transpose((2, 0, 1))
    
    if normalize:
        src_image = (src_image.astype(np.float) / 255) * 2.0 - 1.0
    
    return src_image
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

    heatmaps = blur_heatmaps(heatmaps)
    return heatmaps


def blur_heatmaps(heatmaps):
    """Blurs heatmaps using GaussinaBlur of defined size"""
    heatmaps_blurred = heatmaps.copy()
    for k in range(len(heatmaps)):
        if heatmaps_blurred[k].max() == 1:
            heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], (51, 51), 3)
            heatmaps_blurred[k] = heatmaps_blurred[k] / heatmaps_blurred[k].max()
    return heatmaps_blurred
'''
    align ty pelvis
    joints: n x 14 x 3, by lsp order
'''
def align_by_pelvis(joints):
    left_id = 3
    right_id = 2
    pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.0
    return joints - torch.unsqueeze(pelvis, dim=1)

def copy_state_dict(cur_state_dict, pre_state_dict, prefix = ''):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None
    
    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            print('copy param {} failed'.format(k))
            continue


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    #shape of kps_mask (1002, 1000, 3)
    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0] #0 for l = 0
        i2 = kps_lines[l][1] #7 for l = 0 check skeliton shape tupple   kps shape [3,17]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def world2cam(world_coord, R, t):
    # print(f'word coordinate shape :{world_coord.shape}')
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2pixel(cam_coord, f, c): #cam_coord shape (17,3)
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
#    print(f'image_coord shape {img_coord.shape}')
    return img_coord


class IoULoss(nn.Module):
    """
    Intersection over Union Loss.
    IoU = Area of Overlap / Area of Union
    IoU loss is modified to use for heatmaps.
    """

    def __init__(self):
        super(IoULoss, self).__init__()
        self.EPSILON = 1e-6

    def _op_sum(self, x):
        return x.sum(-1).sum(-1)

    def forward(self, y_pred, y_true):
        inter = self._op_sum(y_true * y_pred)
        union = (
            self._op_sum(y_true ** 2)
            + self._op_sum(y_pred ** 2)
            - self._op_sum(y_true * y_pred)
        )
        iou = (inter + self.EPSILON) / (union + self.EPSILON)
        iou = torch.mean(iou)
        return 1 - iou


def heatmaps_to_coordinates(heatmaps):
    """
    Heatmaps is a numpy array
    Its size - (batch_size, n_keypoints, img_size, img_size)
    """
    batch_size = heatmaps.shape[0]
    sums = heatmaps.sum(axis=-1).sum(axis=-1)
    sums = np.expand_dims(sums, [2, 3])
    normalized = heatmaps / sums
    x_prob = normalized.sum(axis=2)
    y_prob = normalized.sum(axis=3)

    arr = np.tile(np.float32(np.arange(0, 128)), [batch_size, 21, 1])
    x = (arr * x_prob).sum(axis=2)
    y = (arr * y_prob).sum(axis=2)
    keypoints = np.stack([x, y], axis=-1)
    return keypoints / 128


if __name__ == '__main__':
    image_path = 'E:/HMR/data/COCO/images/train-valid2017/000000000009.jpg'
    lt = np.array([-10, -10], dtype = np.float)
    rb = np.array([10,10], dtype = np.float)
    print(crop_image(image_path, 45, lt, rb, [1, 1, 1, 1], None))

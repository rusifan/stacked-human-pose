from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import numpy as np
import cv2

import _init_paths
from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
import pose.datasets as datasets
import pose.losses as losses
import torch.nn as nn
from scipy.signal import find_peaks
# from pose.datasets.human36_dataloader import hum36m_dataloader


DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1966, 0.1734, 0.1836]

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x


def soft_argmax(voxels):
	"""
	Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
	Return: 3D coordinates in shape (batch_size, channel, 3)
	"""
	assert voxels.dim()==5
	# alpha is here to make the largest element really big, so it
	# would become very close to 1 after softmax
	alpha = 1000.0 
	N,C,H,W,D = voxels.shape
	soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
	soft_max = soft_max.view(voxels.shape)
	indices_kernel = torch.arange(start=0,end=H*W*D).unsqueeze(0).to('cuda')
	indices_kernel = indices_kernel.view((H,W,D))
	conv = soft_max*indices_kernel
	indices = conv.sum(2).sum(2).sum(2)
	z = indices%D
	y = (indices/D).floor()%W
	x = (((indices/D).floor())/W).floor()%H
	coords = torch.stack([y,x,z],dim=2)
	return coords[:,:,:2]/64.0

def generate_keypoints(heatmap, threshold=0.5):
    keypoints = np.zeros((17, 2))
    for i in range(17):
        peaks, _ = find_peaks(heatmap[i].ravel(), height=threshold)
        if len(peaks) > 0:
            y, x = np.unravel_index(peaks, heatmap[i].shape)
            max_peak = np.argmax(heatmap[i, y, x])
            keypoints[i] = [x[max_peak], y[max_peak]]
        else:
            keypoints[i] = [0, 0]
    return keypoints / 64.0
# get model names and dataset names
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))


# init global variables
best_acc = 0
idx = []

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33

def main(args):
    global best_acc
    global idx

    # idx is the index of joints used to compute accuracy
    if args.dataset in ['mpii', 'lsp']:
        idx = [1,2,3,4,5,6,11,12,15,16]
    elif args.dataset == 'coco':
        idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    else:
        print("Unknown dataset: {}".format(args.dataset))
        assert False

    # create checkpoint dir
    # if not isdir(args.checkpoint):
    #     mkdir_p(args.checkpoint)

    # create model
    njoints = datasets.__dict__[args.dataset].njoints

    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks,
                                       num_blocks=args.blocks,
                                       num_classes=njoints,
                                       resnet_layers=args.resnet_layers)

    # model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = losses.JointsMSELoss().to(device)

    if args.solver == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    elif args.solver == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )
    else:
        print('Unknown solver: {}'.format(args.solver))
        assert False

    # optionally resume from a checkpoint
    title = args.dataset + ' ' + args.arch
    # if args.resume:
    #     if isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc = checkpoint['best_acc']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #         logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))
    # else:
    #     logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
    #     logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss',
    #                       'Train Acc', 'Val Acc'])

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters())/1000000.0))

    # create data loader
    # train_dataset = datasets.__dict__[args.dataset](is_train=True, **vars(args))
    # # r = train_dataset.__getitem__(0)
    # # print(r[0].shape) torch.Size([3, 256, 256])
    # # print(r[1].shape) torch.Size([16, 64, 64])
    # # import pdb;pdb.set_trace()
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.train_batch, shuffle=True,
    #     num_workers=args.workers, pin_memory=True
    # )

    val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    def convert_image_by_pixformat_normalize(src_image, pix_format, normalize):
        if pix_format == 'NCHW':
            src_image = src_image.transpose((2, 0, 1))
        
        if normalize:
            src_image = (src_image.astype(np.float) / 255) * 2.0 - 1.0
        
        return src_image
#q: unzip files in ubuntu terminal?
#a: unzip -q -d /netscratch/nafis/human-pose/pytorch-pose/data/mpii/mpii_human_pose_v1_u12_1.zip -d /netscratch/nafis/human-pose/pytorch-pose/load_check
    lr = args.lr
    filename = '/netscratch/nafis/human-pose/pytorch-pose/results/img/order_fix/'
    model.eval()

    # checkpoint = torch.load('/netscratch/nafis/human-pose/pytorch-pose/load_check/mpii/hg_s8_b1/checkpoint.pth.tar')
    # checkpoint = torch.load("/netscratch/nafis/human-pose/pytorch-pose/results/stacked4_14kps/model_35.pth")
    # checkpoint = torch.load("/netscratch/nafis/human-pose/pytorch-pose/results/stacked4_16kps/model_35.pth")
    checkpoint = torch.load("/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/stacked8_16kps_fix/model_42.pth") #mpii only   

    # /netscratch/nafis/human-pose/real_time_pose/results/delete/res_test_6.jpg
    # model.load_state_dict(checkpoint['state_dict'])
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    # for k, v in checkpoint['state_dict'].items():
    for k, v in checkpoint.items():
        # import pdb;pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    image_name = '/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/test_img/test.png' #for test_img
    cvimg = cv2.imread(image_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.resize(cvimg, (256, 256))
    # img = img.transpose(2, 0, 1)
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    img /= 255
    img = torch.from_numpy(img).float().cuda()
    img = color_normalize(img, DATASET_MEANS, DATASET_STDS)
    img = img.unsqueeze(0)
    # r = val_dataset.__getitem__(100)
    # output = model(r[0].unsqueeze(0).to(device))
    # gt_heatmap = r[1]
    r = val_dataset.__getitem__(100)
    # output = model(r[0].unsqueeze(0).to(device))
    output = model(img)
    gt_heatmap = r[1]
    image_path = r[2]['img_path']
    kps = r[2]['pts']
    # img = cv2.imread(image_path)
    output = output[-1].squeeze().cpu().detach().numpy()
    plt.imshow(output.sum(axis=0))
    plt.savefig('/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/vis/only_sthg_heatmap_test_1.png')
    plt.clf()

    #plot the keypoints on the images
    # for i in range(16):
    #     tmp = cv2.circle(cvimg, (int(kps[i][0]), int(kps[i][1])), 4, (0, 0, 255), -1)
    # cv2.imwrite(filename + f'order{i}.jpg', tmp)
    
    # for human 36
    # annotation_path_train = "annotation_body3d/fps25/h36m_train.npz"
    # annotation_path_test = "annotation_body3d/fps25/h36m_test.npz"
    # root_data_path = "/netscratch/nafis/human-pose/dataset_human36_nos7_f25"
    # train_dataset = hum36m_dataloader(root_data_path, annotation_path_train, True, [1.1, 2.0], False, 5, flip_prob = 1)
    # r = train_dataset.__getitem__(0)
    # kps = r['kp_2d_org']
    # image_path = r['img_name']
    # filename = '/netscratch/nafis/human-pose/pytorch-pose/results/img/human_order/'
    # img = cv2.imread(image_path)
    # for i in range(16):
    #     tmp = cv2.circle(cv2.imread(image_path), (int(kps[i][0]), int(kps[i][1])), 4, (0, 0, 255), -1)
    #     cv2.imwrite(filename + f'order{i}.jpg', tmp)

    #image load and convert to tensor and resize to 256x256
    # import torchvision.transforms as transforms
    # import PIL.Image as Image
    # img = Image.open('/netscratch/nafis/human-pose/real_time_pose/results/delete/res_test_6.jpg')
    # # img = Image.open('/netscratch/nafis/human-pose/pytorch-pose/new_img/test.png')
    # # img = cv2.imread('/netscratch/nafis/human-pose/pytorch-pose/new_img/test.png')
    # img = img.resize((256,256))
    # # input_image = cv2.resize(img, (256,256), interpolation = cv2.INTER_CUBIC)
    # # input_image = torch.from_numpy(convert_image_by_pixformat_normalize(input_image, 'NCHW', True)).float()

    # # transform = transforms.Compose([transforms.PILToTensor()])

    # # # img = transform(img)
    # # import pdb;pdb.set_trace()
    # img = transforms.ToTensor()(img)
    # # img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    # #                            std=[0.229, 0.224, 0.225])(img)
    # output = model(img.unsqueeze(0).to(device))
    # # output = model(input_image.unsqueeze(0).to(device))
    # import matplotlib.pyplot as plt
    # plt.imshow(gt_heatmap.sum(axis=0))
    # plt.savefig(filename + 'gt.jpg')
    # plt.clf()
    # # import pdb;pdb.set_trace()
    # # output = output[-1]..detach().cpu().numpy()
    # output = output[-1].unsqueeze(4)
    # key_points = soft_argmax(output)
    # output = output.squeeze().detach().cpu().numpy()
    # key_points = key_points.squeeze().detach().cpu().numpy()
    # #key_points is an array of 16x2 (x,y) coordinates add one row of zeros at the 7th index
    # key_points = np.insert(key_points, 7, 0, axis=0)
    # import pdb;pdb.set_trace()
    # plt.imshow(output.sum(axis=0))
    # plt.savefig(filename + 'pred_test.jpg')
    # plt.clf()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset setting
    parser.add_argument('--dataset', metavar='DATASET', default='mpii',
                        choices=dataset_names,
                        help='Datasets: ' +
                            ' | '.join(dataset_names) +
                            ' (default: mpii)')
    parser.add_argument('--image-path', default='', type=str,
                        help='path to images')
    parser.add_argument('--anno-path', default='', type=str,
                        help='path to annotation (json)')
    parser.add_argument('--year', default=2014, type=int, metavar='N',
                        help='year of coco dataset: 2014 (default) | 2017)')
    parser.add_argument('--inp-res', default=256, type=int,
                        help='input resolution (default: 256)')
    parser.add_argument('--out-res', default=64, type=int,
                    help='output resolution (default: 64, to gen GT)')

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: hg)')
    parser.add_argument('-s', '--stacks', default=8, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('--resnet-layers', default=50, type=int, metavar='N',
                        help='Number of resnet layers',
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    # Training strategy
    parser.add_argument('--solver', metavar='SOLVER', default='rms',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--target-weight', dest='target_weight',
                        action='store_true',
                        help='Loss with target_weight')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--scale-factor', type=float, default=0.25,
                        help='Scale factor (data aug).')
    parser.add_argument('--rot-factor', type=float, default=0,
                        help='Rotation factor (data aug).')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--snapshot', default=0, type=int,
                        help='save models for every #snapshot epochs (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')


    main(parser.parse_args())

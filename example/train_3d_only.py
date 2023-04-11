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
import wandb
from pose.datasets.human36_dataloader import hum36m_dataloader
from tqdm import tqdm
from pose.models.hourglass import *
from model.modulated_gcn import ModulatedGCN
import numpy as np
import torch.nn as nn

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

# get model names and dataset names
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


# init global variables
best_loss = 10000
idx = []

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33

def main():
    global best_loss
    global idx
    train_batch = 128
    test_batch = 32
    workers = 1
    epochs = 100
    schedule = [60,90]
    gamma = 0.1
    sigma_decay = 0
    debug = False
    flip = False
    train_flag = True
    test_flag = True
    wandb_flag = True
    annotation_path_train = "annotation_body3d/fps25/h36m_train.npz"
    annotation_path_test = "annotation_body3d/fps25/h36m_test.npz"
    root_data_path = "/netscratch/nafis/human-pose/dataset_human36_nos7_f25"
    # idx is the index of joints used to compute accuracy
    # if args.dataset in ['mpii', 'lsp']:
    #     idx = [1,2,3,4,5,6,11,12,15,16]
    # elif args.dataset == 'coco':
    #     idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    # else:
    #     print("Unknown dataset: {}".format(args.dataset))
    #     assert False

    # create checkpoint dir
    # if not isdir(args.checkpoint):
    #     mkdir_p(args.checkpoint)
    
    # create model
    njoints = 16
    criterion_l1 = nn.L1Loss(size_average=True).cuda()
    # print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    # model = models.__dict__[args.arch](num_stacks=args.stacks,
    #                                    num_blocks=args.blocks,
    #                                    num_classes=njoints,
    #                                    resnet_layers=args.resnet_layers)
    # model = HourglassNet(1,num_stacks=4, num_blocks=1, num_classes=17)
    model = hg(num_stacks=4, num_blocks=1, num_classes=16)
    checkpoint = torch.load("/netscratch/nafis/human-pose/pytorch-pose/results/stacked4_16kps/model_35.pth")

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    # for k, v in checkpoint['state_dict'].items():
    for k, v in checkpoint.items():
        # import pdb;pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model = torch.nn.DataParallel(model).to(device)
    adj = np.load('/netscratch/nafis/human-pose/Modulated-GCN/Modulated_GCN/Modulated-GCN_benchmark/results/adj_4_16.npy')
    adj = torch.from_numpy(adj).float().to(device)
    mgcn = ModulatedGCN(adj, 384, num_layers=2, p_dropout=0, nodes_group=None).to('cuda')
    if wandb_flag:
        wandb.login()
        wandb.init(project="mpii_stacked", entity="nafisur")
        wandb.run.name = "stack_4_3donly"
        wandb.run.save()
        wandb.watch(mgcn)
    # define loss function (criterion) and optimizer
    criterion = losses.JointsMSELoss().to(device)

    # if args.solver == 'rms':
    optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=2.5e-4,
                                        momentum=0,
                                        weight_decay=0)
    # elif args.solver == 'adam':
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=2.5e-4,
        )
    # else:
    #     print('Unknown solver: {}'.format(args.solver))
    #     assert False

    # optionally resume from a checkpoint
    # title = 'juman3.6m' + ' ' + args.arch
    title = 'juman3.6m'
    resume = False 
    if resume:
        if isfile(resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join('./checkpoint/h36m/test', 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss',
                          'Train Acc', 'Val Acc'])

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters())/1000000.0))

    # create data loader
    # train_dataset = datasets.__dict__[args.dataset](is_train=True, **vars(args))
    train_dataset = hum36m_dataloader(root_data_path, annotation_path_train, True, [1.1, 2.0], False, 5, flip_prob = 1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch, shuffle=True,
        num_workers=workers, pin_memory=True
    )

    # val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_dataset = hum36m_dataloader(root_data_path, annotation_path_test, True, [1.1, 2.0], False, 5, flip_prob = 1)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=test_batch, shuffle=False,
        num_workers=workers, pin_memory=True
    )

    # evaluation only


    # train and eval
    lr = 2.5e-4
    for epoch in range(epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, schedule, gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if sigma_decay > 0:
            train_loader.dataset.sigma *=  sigma_decay
            val_loader.dataset.sigma *=  sigma_decay

        # train for one epoch
        if train_flag:
            train_loss = train(train_loader, model,mgcn, criterion_l1, optimizer,
                                      debug, flip)
        if wandb_flag:
            wandb.log({"train_loss": train_loss})
        # evaluate on validation set
        valid_loss, predictions = validate(val_loader, model,mgcn, criterion_l1,
                                                  njoints, debug, flip)
        if wandb_flag:
            wandb.log({"valid_loss": valid_loss})
        # append logger file
        train_acc, valid_acc = 0, 0
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        if valid_loss < best_loss:
            torch.save(model.state_dict(), f'/netscratch/nafis/human-pose/pytorch-pose/results/human36_3donly/model_{epoch}.pth')
        # is_best = valid_acc > best_acc
            best_loss = min(valid_loss, best_loss)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': model.state_dict(),
        #     'best_acc': best_acc,
        #     'optimizer' : optimizer.state_dict(),
        # }, predictions, is_best, checkpoint=args.checkpoint, snapshot=args.snapshot)
        # torch.save(model.state_dict(), f'/netscratch/nafis/human-pose/pytorch-pose/results/model_{epoch}.pth')
    logger.close()

    # logger.plot(['Train Acc', 'Val Acc'])
    # savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(train_loader, model,mgcn, criterion, optimizer, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.eval()
    mgcn.train()
    end = time.time()

    gt_win, pred_win = None, None
    bar = Bar('Train', max=len(train_loader))
    # for i, (input, target, meta) in enumerate(tqdm(train_loader,0)):
    for i, data in enumerate(tqdm(train_loader,0)):
    # for i, (input, target, meta) in enumerate(train_loader): 3tqdm added
        # measure data loading time
        data_time.update(time.time() - end)
        input = data['image'].to('cuda')
        target = data['heatmap'].to('cuda')
        kps_3d = data['kp_3d_u'].to('cuda')
        ratio_x = data['ratio_x']
        ratio_y = data['ratio_y']
        left_top = data['leftTop']
        # import pdb; pdb.set_trace()
        input, target = input.to(device), target.to(device, non_blocking=True)
        # torch ones of shape [batch_size, 14, 1]
        target_weight = torch.ones(target.size(0), target.size(1), 1)
        target_weight = target_weight.to(device, non_blocking=True)

        # compute output
        # import pdb; pdb.set_trace()
        input = input.permute(0,3,1,2)
        output = model(input)
        output = output[-1]
        N = input.shape[0]
        output = output.unsqueeze(4)
        key_points = soft_argmax(output)
        # import pdb; pdb.set_trace()
        for i in range(N):
            key_points[i,:,0] /= ratio_x[i]
            key_points[i,:,1] /= ratio_y[i]
        for i in range(N):
            key_points[i,:,0] += left_top[0][i]
            key_points[i,:,1] += left_top[1][i]
        for i in range(N):
            key_points[i,:,0] *= ratio_x[i]
            key_points[i,:,1] *= ratio_y[i]
        key_points = key_points.view(N, -1, 16, 2, 1).permute(0, 3, 1, 2, 4).to('cuda')
        pred_key_3d = mgcn(key_points)
        pred_key_3d = pred_key_3d.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, 16, 3)
        kps_3d = kps_3d.unsqueeze(1)
        # import pdb;pdb.set_trace()

        loss = (1 - 0.01)*mpjpe(pred_key_3d, kps_3d)+criterion(pred_key_3d, kps_3d)
        # if type(output) == list:  # multiple output
        #     loss = 0
        #     for o in output:
        #         loss += criterion(o, target, target_weight)
        #     output = output[-1]
        # else:  # single output
        #     loss = criterion(output, target, target_weight)
        # acc = accuracy(output, target, idx)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        # acces.update(acc[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=acces.avg
                    )
        bar.next()

    bar.finish()
    return losses.avg


def validate(val_loader, model,mgcn, criterion, num_classes, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    model.eval()
    mgcn.eval()
    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader,0)):
        # for i, (input, target, meta) in enumerate(val_loader): 3tqdm added
            # measure data loading time
            data_time.update(time.time() - end)
            input = data['image'].to('cuda')
            target = data['heatmap'].to('cuda')
            kps_3d = data['kp_3d_u'].to('cuda')
            ratio_x = data['ratio_x']
            ratio_y = data['ratio_y']
            left_top = data['leftTop']
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_weight = torch.ones(target.size(0), target.size(1), 1)
            target_weight = target_weight.to(device, non_blocking=True)

            # target_weight = meta['target_weight'].to(device, non_blocking=True)
            input = input.permute(0,3,1,2)
            # compute output
            output = model(input)
            output = output[-1]
            N = input.shape[0]
            output = output.unsqueeze(4)

            key_points = soft_argmax(output)
            for i in range(N):
                key_points[i,:,0] /= ratio_x[i]
                key_points[i,:,1] /= ratio_y[i]
            for i in range(N):
                key_points[i,:,0] += left_top[0][i]
                key_points[i,:,1] += left_top[1][i]
            for i in range(N):
                key_points[i,:,0] *= ratio_x[i]
                key_points[i,:,1] *= ratio_y[i]
            key_points = key_points.view(N, -1, 16, 2, 1).permute(0, 3, 1, 2, 4).to('cuda')
            pred_key_3d = mgcn(key_points)
            pred_key_3d = pred_key_3d.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, 16, 3)
            kps_3d = kps_3d.unsqueeze(1)
            loss = mpjpe(pred_key_3d, kps_3d)
            # score_map = output[-1].cpu() if type(output) == list else output.cpu()
            # if flip:
            #     flip_input = torch.from_numpy(fliplr(input.clone().numpy())).float().to(device)
            #     flip_output = model(flip_input)
            #     flip_output = flip_output[-1].cpu() if type(flip_output) == list else flip_output.cpu()
            #     flip_output = flip_back(flip_output)
            #     score_map += flip_output



            # if type(output) == list:  # multiple output
            #     loss = 0
            #     for o in output:
            #         loss += criterion(o, target, target_weight)
            #     output = output[-1]
            # else:  # single output
            #     loss = criterion(output, target, target_weight)

            # acc = accuracy(score_map, target.cpu(), idx)

            # generate predictions
            # preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]
            losses.update(loss.item(), input.size(0))


            # acces.update(acc[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                        batch=i + 1,
                        size=len(val_loader),
                        data=data_time.val,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        acc=acces.avg
                        )
            bar.next()

        bar.finish()
    return losses.avg, predictions

if __name__ == '__main__':

    main()

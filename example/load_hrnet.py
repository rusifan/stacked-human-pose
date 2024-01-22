from __future__ import print_function, absolute_import
from collections import OrderedDict

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

import _init_paths
from pose.models.config import _C as cfg
from pose.models.HRnet_model import get_pose_net
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
from tqdm import tqdm


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))


# init global variables
best_loss = 10000
idx = []

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33

def main(args):
    global best_loss
    global idx

    wandb_flag = True
    # idx is the index of joints used to compute accuracy
    if args.dataset in ['mpii', 'lsp']:
        idx = [1,2,3,4,5,6,11,12,15,16]
    elif args.dataset == 'coco':
        idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    else:
        print("Unknown dataset: {}".format(args.dataset))
        assert False

    # create checkpoint dir
   
    njoints = datasets.__dict__[args.dataset].njoints

    # create model
    model = get_pose_net(cfg, is_train=True, njoints=njoints)
    # load pre-trained model
    # state_dict = torch.load('/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/full_net16kps/model_11.pth') #16kps checkpoint pre trained on mpii
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    model = torch.nn.DataParallel(model).cuda()

    if wandb_flag:
        wandb.login()
        wandb.init(project="mpii_human", entity="nafisur")
        wandb.run.name = "hrnet_mpii"
        wandb.run.save()
        wandb.watch(model)
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
    if args.resume:
        if isfile(args.resume):
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
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss',
                          'Train Acc', 'Val Acc'])

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters())/1000000.0))

    # create data loader
    train_dataset = datasets.__dict__[args.dataset](is_train=True, **vars(args))
    # r = train_dataset.__getitem__(0)
    # print(r[0].shape) #torch.Size([3, 256, 256])
    # print(r[1].shape) #torch.Size([16, 64, 64])
    # import pdb;pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    # import pdb;pdb.set_trace()

    # evaluation only
    if args.evaluate:
        print('\nEvaluation only')
        loss, acc, predictions = validate(val_loader, model, criterion, njoints,
                                          args.debug, args.flip)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    # train and eval
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *=  args.sigma_decay
            val_loader.dataset.sigma *=  args.sigma_decay

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer,
                                      args.debug, args.flip)
        print(f'train_loss: {train_loss}')
        if wandb_flag:
            wandb.log({"train_loss": train_loss})
        # evaluate on validation set
        valid_loss, predictions = validate(val_loader, model, criterion,
                                                  njoints, args.debug, args.flip)
        print(f'valid_loss: {valid_loss}')
        if wandb_flag:
            wandb.log({"testing error 2d": valid_loss})
        # append logger file
        train_acc, valid_acc = 0, 0
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        if valid_loss < best_loss:
            torch.save(model.state_dict(), f'/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/hrnet/model_{epoch}.pth')
        # is_best = valid_acc > best_acc
        # /netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/train_fullnet_mpii_only/model_0.pth
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


def train(train_loader, model, criterion, optimizer, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    gt_win, pred_win = None, None
    bar = Bar('Train', max=len(train_loader))
    for i, (input, target, meta) in enumerate(tqdm(train_loader,0)):
    # for i, (input, target, meta) in enumerate(train_loader): 3tqdm added
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device, non_blocking=True)
        target_weight = meta['target_weight'].to(device, non_blocking=True)
        left_top = meta['left_top']
        ratio_x = meta['ratio_x'].to('cuda')
        ratio_y = meta['ratio_y'].to('cuda')
        # import pdb;pdb.set_trace()

        # compute output
        # output = model(input)
        # predicted_out3d, output = model(input,left_top,ratio_x,ratio_y)
        output = model(input)
        if type(output) == list:  # multiple output
            loss = 0
            for o in output:
                loss += criterion(o, target, target_weight)
            output = output[-1]
        else:  # single output
            loss = criterion(output, target, target_weight)
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


def validate(val_loader, model, criterion, num_classes, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))
    with torch.no_grad():
        for i, (input, target, meta) in enumerate(tqdm(val_loader,0)):
        # for i, (input, target, meta) in enumerate(val_loader): 3tqdm added
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_weight = meta['target_weight'].to(device, non_blocking=True)
            left_top = meta['left_top']
            ratio_x = meta['ratio_x'].to('cuda')
            ratio_y = meta['ratio_y'].to('cuda')
            # compute output
            # output = model(input)
            output = model(input)

            score_map = output[-1].cpu() if type(output) == list else output.cpu()
            if flip:
                flip_input = torch.from_numpy(fliplr(input.clone().numpy())).float().to(device)
                flip_output = model(flip_input)
                flip_output = flip_output[-1].cpu() if type(flip_output) == list else flip_output.cpu()
                flip_output = flip_back(flip_output)
                score_map += flip_output



            if type(output) == list:  # multiple output
                loss = 0
                for o in output:
                    loss += criterion(o, target, target_weight)
                output = output[-1]
            else:  # single output
                loss = criterion(output, target, target_weight)

            # acc = accuracy(score_map, target.cpu(), idx)

            # generate predictions
            preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure accuracy and record loss
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
    parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=32, type=int, metavar='N',
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
    # parser.add_argument( '--wandb_flag', dest='debug', action='store_true',
    #                     help='show intermediate results')

    main(parser.parse_args())
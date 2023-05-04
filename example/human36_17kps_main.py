from __future__ import print_function, absolute_import

import os
import time
from pathlib import Path
from typing import Optional
from collections import OrderedDict

import tqdm
import wandb
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import _init_paths

import pose.losses as losses
import pose.models as models
import pose.datasets as datasets
from pose.datasets.human17kps import hum36m_dataloader
from pose.models.hourglass import *
from pose.utils.evaluation import accuracy, AverageMeter
from pose.utils.imutils import batch_with_heatmap
from pose.utils.logger import Logger, savefig
from pose.utils.misc import save_pred, adjust_learning_rate
from pose.utils.osutils import isfile, join
from pose.utils.transforms import fliplr, flip_back
from pose.utils.visualize import show, show_heatmap

# get model names and dataset names
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


def load_model(weight_path: Optional[Path]):
    model = hg(num_stacks=4, num_blocks=1, num_classes=17)
    if weight_path is not None:
        checkpoint = torch.load(str(weight_path))

        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

    model = torch.nn.DataParallel(model).to(device)
    return model


def main():
    global best_loss
    global idx
    train_batch = 16
    test_batch = 16
    workers = 1
    epochs = 100
    schedule = [60, 90]
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
    # root_data_path = "./data/human3.6/"
    # idx is the index of joints used to compute accuracy
    # if args.dataset in ['mpii', 'lsp']:
    #     idx = [1,2,3,4,5,6,11,12,15,16]
    # elif args.dataset == 'coco':
    #     idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # else:
    #     print("Unknown dataset: {}".format(args.dataset))
    #     assert False

    # create checkpoint dir
    # if not isdir(args.checkpoint):
    #     mkdir_p(args.checkpoint)

    # create model
    njoints = 17

    # print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    # model = models.__dict__[args.arch](num_stacks=args.stacks,
    #                                    num_blocks=args.blocks,
    #                                    num_classes=njoints,
    #                                    resnet_layers=args.resnet_layers)
    # model = HourglassNet(1,num_stacks=4, num_blocks=1, num_classes=17)

    # model = load_model(weight_path=Path("./checkpoint/mpii/model_35.pth"))
    checkpoint = '/netscratch/nafis/human-pose/pytorch-pose/results/stacked4_16kps/model_35.pth'
    model = load_model(weight_path=None)
    if wandb_flag:
        wandb.login()
        wandb.init(project="mpii_human", entity="nafisur")
        wandb.run.name = "stack_4_17kps"
        wandb.run.save()
        wandb.watch(model)
    # define loss function (criterion) and optimizer
    criterion = losses.JointsMSELoss().to(device)

    # if args.solver == 'rms':
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=2.5e-4,
                                    momentum=0,
                                    weight_decay=0)

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
        log_path = Path('./checkpoint/h36m/test/log.txt')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = Logger(str(log_path), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss',
                          'Train Acc', 'Val Acc'])

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # create data loader
    # train_dataset = datasets.__dict__[args.dataset](is_train=True, **vars(args))
    train_dataset = hum36m_dataloader(root_data_path, annotation_path_train, True, [1.1, 2.0], False, 5, flip_prob=1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch, shuffle=True,
        num_workers=workers, pin_memory=True
    )

    # val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_dataset = hum36m_dataloader(root_data_path, annotation_path_test, True, [1.1, 2.0], False, 5, flip_prob=1)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=test_batch, shuffle=False,
        num_workers=workers, pin_memory=True
    )

    # evaluation only
    evaluate = False
    if evaluate:
        print('\nEvaluation only')
        loss, acc, predictions = validate(val_loader, model, criterion, njoints,
                                          args.debug, args.flip)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    # validate(val_loader, model, criterion, njoints, debug, flip)

    # train and eval
    lr = 2.5e-4
    for epoch in range(epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, schedule, gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if sigma_decay > 0:
            train_loader.dataset.sigma *= sigma_decay
            val_loader.dataset.sigma *= sigma_decay

        # train for one epoch
        if train_flag:
            train_loss = train(train_loader, model, criterion, optimizer,
                               debug, flip)
            print('Train Loss: %.8f' % (train_loss))
        if wandb_flag:
            wandb.log({"train_loss": train_loss})
        # evaluate on validation set
        valid_loss, predictions = validate(val_loader, model, criterion,
                                           njoints, debug, flip)
        print('Valid Loss: %.8f' % (valid_loss))
        if wandb_flag:
            wandb.log({"valid_loss": valid_loss})
        # append logger file
        train_acc, valid_acc = 0, 0
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        if valid_loss < best_loss:
            result_path = Path(f'results/pre_mpii_human/model_{epoch}.pth')
            result_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(result_path))
            # is_best = valid_acc > best_acc
            best_loss = min(valid_loss, best_loss)
    logger.close()

    logger.plot(['Train Acc', 'Val Acc'])
    savefig('log.png')


def train(train_loader, model, criterion, optimizer, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    gt_win, pred_win = None, None
    bar = tqdm.tqdm(total=len(train_loader), desc='Train')
    # for i, (input, target, meta) in enumerate(tqdm(train_loader,0)):
    for i, data in enumerate(train_loader):
        # for i, (input, target, meta) in enumerate(train_loader): 3tqdm added
        # measure data loading time
        data_time.update(time.time() - end)
        input = data['image']
        target = data['heatmap']
        # import pdb; pdb.set_trace()
        input, target = input.to(device), target.to(device, non_blocking=True)
        # torch ones of shape [batch_size, 14, 1]
        target_weight = torch.ones(target.size(0), target.size(1), 1)
        target_weight = target_weight.to(device, non_blocking=True)

        # compute output
        # import pdb; pdb.set_trace()
        # input = input.permute(0, 3, 1, 2)
        output = model(input)

        if type(output) == list:  # multiple output
            loss = 0
            for o in output:
                loss += criterion(o, target, target_weight)
            output = output[-1]
        else:  # single output
            loss = criterion(output, target, target_weight)
        acc = accuracy(output, target, idx)

        if debug:  # visualize groundtruth and predictions
            gt_batch_img = batch_with_heatmap(input, target)
            pred_batch_img = batch_with_heatmap(input, output)
            if not gt_win or not pred_win:
                ax1 = plt.subplot(121)
                ax1.title.set_text('Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                ax2 = plt.subplot(122)
                ax2.title.set_text('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acces.update(acc[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        description = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.total,
            # eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg
        )
        bar.set_description(description)
        bar.update()

    bar.close()
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
    bar = tqdm.tqdm(desc='Eval', total=len(val_loader))
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # for i, (input, target, meta) in enumerate(val_loader): 3tqdm added
            # measure data loading time
            data_time.update(time.time() - end)
            input = data['image']
            target = data['heatmap']
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_weight = torch.ones(target.size(0), target.size(1), 1)
            target_weight = target_weight.to(device, non_blocking=True)

            # compute output
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
            else:  # single output
                loss = criterion(output, target, target_weight)

            acc = accuracy(score_map, target.cpu(), idx)

            # generate predictions
            # preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]

            if debug:
                gt_batch_img = batch_with_heatmap(input, target)
                pred_batch_img = batch_with_heatmap(input, score_map)
                if not gt_win or not pred_win:
                    plt.subplot(121)
                    gt_win = plt.imshow(gt_batch_img)
                    plt.subplot(122)
                    pred_win = plt.imshow(pred_batch_img)
                else:
                    gt_win.set_data(gt_batch_img)
                    pred_win.set_data(pred_batch_img)
                plt.pause(.05)
                plt.draw()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            acces.update(acc[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            description = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                batch=i + 1,
                size=len(val_loader),
                data=data_time.val,
                bt=batch_time.val,
                total=bar.total,
                # eta=bar.eta_td,
                loss=losses.avg,
                acc=acces.avg
            )
            bar.set_description(description)
            bar.update()

        # show(input[0] + 0.5, output[0][0], target[0])
        # show_heatmap(output[0][0], target[0])

        bar.close()
    return losses.avg, predictions


if __name__ == '__main__':
    main()

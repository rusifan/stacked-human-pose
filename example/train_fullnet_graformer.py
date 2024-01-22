from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import os
import time
from collections import OrderedDict
import _init_paths
import tqdm
import tqdm
import wandb
import matplotlib.pyplot as plt

import numpy as np
import pose.losses as losses
import pose.models as models
import pose.datasets as datasets
from pose.datasets.human36_dataloader import hum36m_dataloader #16kps
from pose.utils.evaluation import accuracy, AverageMeter
from pose.utils.imutils import batch_with_heatmap
from pose.utils.logger import Logger, savefig
from pose.utils.misc import save_pred, adjust_learning_rate
from pose.utils.osutils import isfile, join
from pose.utils.transforms import fliplr, flip_back
from pose.utils.visualize import show, show_heatmap 
# from mynet import MyNet
# from mynet_sgcn import MyNet #sgcn
from net_graformer import MyNet #graformer
# from mynet_8hg_4mgcn import MyNet #8hg_4mgcn
# from mynet_2hg_2mgcn import MyNet #2hg_2mgcn

best_loss = 1e10
idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def main():
    global best_loss
    global idx
    train_batch = 32
    test_batch = 32
    workers = 1
    epochs = 100
    lr = 2.5e-4
    # schedule = [60, 90]
    schedule = [ 30, 60,90]
    # gamma = 0.1
    gamma = 0.3
    sigma_decay = 0
    njoints = 16
    debug = False
    flip = False
    train_flag = True
    test_flag = True
    wandb_flag = True
    run_name = "stgh4_GraFormer_dim_64"

    # # full dataset 
    # annotation_path_train = "annotation_body3d/fps25/h36m_train.npz"
    # annotation_path_test = "annotation_body3d/fps25/h36m_test.npz"
    # root_data_path = "/netscratch/nafis/human-pose/dataset_human36_nos7_f25"

    #smaller dataset
    # annotation_path_train = "annotation_body3d/fps25/h36m_smallertrain.npz"
    # annotation_path_test = "annotation_body3d/fps25/h36m_smallertest.npz"
    # root_data_path = "/netscratch/nafis/human-pose/pose_dataset_h36_small"

    #one fps dataset
    # annotation_path_train = "annotation_body3d/fps1/h36m_train.npz"
    # annotation_path_test = "annotation_body3d/fps1/h36m_test.npz"
    # root_data_path = "/netscratch/nafis/human-pose/dataset_hum36m_1f1s"

    #5fps dataset
    annotation_path_train = "annotation_body3d/fps5/h36m_train.npz"
    annotation_path_test = "annotation_body3d/fps5/h36m_test.npz"
    root_data_path = "/netscratch/nafis/human-pose/dataset_hum36m_10f1s"

    log_path = Path('./checkpoint/h36m/test/log.txt')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = Logger(str(log_path), title='human3.6')
    logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss',
                        'Train Acc', 'Val Acc'])
    # create model
    print("=> creating model ...")
    adj = np.load('/netscratch/nafis/human-pose/Modulated-GCN/Modulated_GCN/Modulated-GCN_benchmark/results/adj_4_16.npy')
    adj = torch.from_numpy(adj).to('cuda')
    model = MyNet(adj, block=2, num_stacks=4) # 4stack and 2 gcn
    # model = MyNet(adj, block=2, num_stacks=8) # 8stack and 2 gcn
    # model = MyNet(adj, block=2, num_stacks=2) # 8stack and 2 gcn
    # model = MyNet(adj, block=4, num_stacks=4) # 8stack and 4 gcn
    model = torch.nn.DataParallel(model).cuda()

    # print size of model in mellions

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]) / 1000000.0))
    #experiment tracking
    if wandb_flag:
        wandb.login()
        wandb.init(project="to_check_accuaracy", entity="nafisur")
        wandb.run.name = run_name
        wandb.run.save()
        wandb.watch(model)
    
    #loss function
    criterion = losses.JointsMSELoss().to('cuda')

    #optimizer
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=2.5e-4,
                                    momentum=0,
                                    weight_decay=0)
    
    #data loader
    print("=> loading data ...")
    train_dataset = hum36m_dataloader(root_data_path, annotation_path_train, True, [1.1, 2.0], False, 5, flip_prob=1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch, shuffle=True,
        num_workers=workers, pin_memory=True
    )

    val_dataset = hum36m_dataloader(root_data_path, annotation_path_test, True, [1.1, 2.0], False, 5, flip_prob=1)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=test_batch, shuffle=False,
        num_workers=workers, pin_memory=True
    )

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
        valid_loss, acc = validate(val_loader, model, criterion,
                                           njoints, epoch, debug, flip)
        print('Valid Loss(mpjpe) in mm: %.8f' % (valid_loss * 1000))
        if wandb_flag:
            wandb.log({"Valid Loss(mpjpe) in mm": valid_loss * 1000})
            wandb.log({"Validation accuracy": acc})
        # append logger file
        train_acc, valid_acc = 0, 0
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        if valid_loss < best_loss:
            result_path = Path(f'results/stgh2_GraFormer_dim_32/model_{epoch}.pth')
            result_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(result_path))
            # is_best = valid_acc > best_acc
            best_loss = min(valid_loss, best_loss)

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
        # measure data loading time
        data_time.update(time.time() - end)
        
        #get the data
        input = data['image']
        target = data['heatmap']
        ratio_x = data['ratio_x'].to('cuda')
        ratio_y = data['ratio_y'].to('cuda')
        left_top = data['leftTop']
        gt_3d = data['kp_3d'].to('cuda')
        N = input.size(0)

        input, target = input.to(device), target.to(device, non_blocking=True)
        target_weight = torch.ones(target.size(0), target.size(1), 1)
        target_weight = target_weight.to(device, non_blocking=True)
        
        # compute output
        predicted_out3d, output = model(input,left_top,ratio_x,ratio_y)
        # import pdb;pdb.set_trace()
        predicted_out3d = predicted_out3d.contiguous().view(N, -1, 16, 3)
        gt_3d = gt_3d.unsqueeze(1)

        # measure accuracy and record loss
        if type(output) == list:  # multiple output
            loss_2d = 0
            for o in output:
                loss_2d += criterion(o, target, target_weight)
            output = output[-1]
        else:  # single output
            loss_2d = criterion(output, target, target_weight)
        loss_3d = mpjpe(predicted_out3d, gt_3d)
        # import pdb;pdb.set_trace()
        loss = 100*loss_2d + loss_3d # equal weight
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

def validate(val_loader, model, criterion, num_classes, epoch, debug=False, flip=True):
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
            ratio_x = data['ratio_x'].to('cuda')
            ratio_y = data['ratio_y'].to('cuda')
            left_top = data['leftTop']
            gt_3d = data['kp_3d'].to('cuda')
            N = input.size(0)

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_weight = torch.ones(target.size(0), target.size(1), 1)
            target_weight = target_weight.to(device, non_blocking=True)

            # compute output
            predicted_out3d, output = model(input,left_top,ratio_x,ratio_y)
            predicted_out3d = predicted_out3d.contiguous().view(N, -1, 16, 3)
            gt_3d = gt_3d.unsqueeze(1)

            score_map = output[-1].cpu() if type(output) == list else output.cpu()
            if flip:
                flip_input = torch.from_numpy(fliplr(input.clone().numpy())).float().to(device)
                flip_output = model(flip_input)
                flip_output = flip_output[-1].cpu() if type(flip_output) == list else flip_output.cpu()
                flip_output = flip_back(flip_output)
                score_map += flip_output

            if type(output) == list:  # multiple output
                loss_2d = 0
                for o in output:
                    loss_2d += criterion(o, target, target_weight)
            else:  # single output
                loss_2d = criterion(output, target, target_weight)
            loss_3d = mpjpe(predicted_out3d, gt_3d)

            acc = accuracy(score_map, target.cpu(), idx)

            # generate predictions
            # preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure accuracy and record loss
            losses.update(loss_3d.item(), input.size(0))
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
        # show(input[0] + 0.5, output[0][0], target[0], Path(f'./results/vis/full_smallerGauss/{epoch}_epoch8stg_schedul.png'))
        # show_heatmap(output[-1][0], target[0], Path(f'./results/vis/heatmap_smallerGauss/{epoch}_epoch8stg_schedul.png'))
        bar.close()
    return losses.avg, acces.avg

if __name__ == '__main__':
    main()

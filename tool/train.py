import os
import time
import random
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import dataset, transform, config
from util.s3dis import S3DIS
from util.scannet import ScanNet
from util.util import AverageMeter, intersectionAndUnionGPU


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointweb.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointweb.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def init():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
    logger.info(args)


def main():
    init()
    if args.arch == 'pointnet_seg':
        from model.pointnet.pointnet import PointNetSeg as Model
    elif args.arch == 'pointnet2_seg':
        from model.pointnet2.pointnet2_seg import PointNet2SSGSeg as Model
    elif args.arch == 'pointweb_seg':
        from model.pointweb.pointweb_seg import PointWebSeg as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes, use_xyz=args.use_xyz)
    if args.sync_bn:
        from util.util import convert_to_syncbn
        convert_to_syncbn(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_epoch, gamma=args.multiplier)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    model = torch.nn.DataParallel(model.cuda())
    if args.sync_bn:
        from lib.sync_bn import patch_replication_callback
        patch_replication_callback(model)
    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    train_transform = transform.Compose([transform.ToTensor()])
    if args.data_name == 's3dis':
        train_data = S3DIS(split='train', data_root=args.train_full_folder, num_point=args.num_point, test_area=args.test_area, block_size=args.block_size, sample_rate=args.sample_rate, transform=train_transform)
        # train_data = dataset.PointData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform)
    elif args.data_name == 'scannet':
        train_data = ScanNet(split='train', data_root=args.data_root, num_point=args.num_point, block_size=args.block_size, sample_rate=args.sample_rate, transform=train_transform)
    elif args.data_name == 'modelnet40':
        train_data = dataset.PointData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform, num_point=args.num_point, random_index=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_workers, pin_memory=True)

    val_loader = None
    if args.evaluate:
        val_transform = transform.Compose([transform.ToTensor()])
        val_data = dataset.PointData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.train_batch_size_val, shuffle=False, num_workers=args.train_workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch)
        epoch_log = epoch + 1
        writer.add_scalar('loss_train', loss_train, epoch_log)
        writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
        writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
        writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if epoch_log % args.save_freq == 0:
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, filename)
            if epoch_log / args.save_freq > 2:
                deletename = args.save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
                os.remove(deletename)
        if args.evaluate:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
            writer.add_scalar('loss_val', loss_val, epoch_log)
            writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
            writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
            writer.add_scalar('allAcc_val', allAcc_val, epoch_log)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
        writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
        writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
        writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        output = model(input)
        loss = criterion(output, target)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()

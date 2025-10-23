import logging
import os
import random
import sys
import numpy as np  # <-- 1. 已添加此 import
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json

from utils import DiceLoss
from scipy.ndimage.interpolation import zoom


# 2. ValGenerator 类 (现在 np 将被正确识别)
class ValGenerator(object):
    """
    一个用于验证集的转换器，只进行缩放和张量转换，不进行随机数据增强。
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            # 使用 order=3 (双三次) 插值处理图像，order=0 (最近邻) 处理标签
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                             transform=transforms.Compose(
                                 [ValGenerator(output_size=[args.img_size, args.img_size])]))

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    history = {
        'train_total_loss': [], 'train_ce_loss': [], 'train_dice_loss': [],
        'val_total_loss': [], 'val_ce_loss': [], 'val_dice_loss': []
    }
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", total=len(train_loader),
                                           leave=False):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # logging.info('Train: iteration : %d/%d, lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (
            #     iter_num, epoch_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
        history['train_total_loss'].append(batch_loss)
        history['train_ce_loss'].append(batch_ce_loss)
        history['train_dice_loss'].append(batch_dice_loss)
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                                   total=len(val_loader), leave=False):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    batch_dice_loss += loss_dice.item()
                    batch_ce_loss += loss_ce.item()

                batch_ce_loss /= len(val_loader)
                batch_dice_loss /= len(val_loader)
                batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
                logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
                    epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
                history['val_total_loss'].append(batch_loss)
                history['val_ce_loss'].append(batch_ce_loss)
                history['val_dice_loss'].append(batch_dice_loss)
                if batch_loss < best_loss:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = batch_loss
                else:
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
    loss_history_path = os.path.join(snapshot_path, "training_losses.json")
    try:
        with open(loss_history_path, 'w') as f:
            json.dump(history, f, indent=4)
        logging.info(f"Loss history saved to {loss_history_path}")
    except Exception as e:
        logging.error(f"Failed to save loss history: {e}")

    writer.close()
    return "Training Finished!"

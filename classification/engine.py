# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    my_softmax = torch.nn.Softmax(dim=1)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        targets = targets.to(device)
        targets = targets.to(torch.int64)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs = outputs.to(device)
            outputs = outputs.to(torch.float32)
            outputs = my_softmax(outputs)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


import torch
import torch.nn as nn

class pF1(nn.Module):
    def __init__(self):
        super(pF1, self).__init__()
        self.tc = nn.Parameter(torch.zeros(1))  ##tc(真値:本物)=tp+fn
        self.tp = nn.Parameter(torch.zeros(1))
        self.fp = nn.Parameter(torch.zeros(1))
        

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tc.data += y_true[y_true==1].size()[0]
        tmp1 = y_true[y_pred == 1]  #y_pred==1のture
        #tmp0 = y_pred[y_true == 0]
        self.tp.data += tmp1[tmp1==1].size()[0]  #tmp1のなかでtrue1
        self.fp.data += tmp1[tmp1!=1].size()[0]  #tmp1のなかでtrue0

    def result(self):
        #print(self.tc.data, self.tp.data, self.fp.data)
        if self.tc == 0 or (self.tp + self.fp) == 0:
            return torch.tensor(0.0)
        else:
            precision = self.tp / (self.tp + self.fp)
            recall = self.tp / self.tc  ##tp+fn=tc
            return 2 * (precision * recall) / (precision + recall)

    def reset_state(self):
        self.tc.data = torch.zeros(1)
        self.tp.data = torch.zeros(1)
        self.fp.data = torch.zeros(1)


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss(torch.tensor([1.0, 10.0], device='cuda'))

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pf1 = pF1()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc2 = accuracy(output, target, topk=(1, 2))
        pf1.update_state(target, output)
        f1_score=pf1.result()

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['pf1'].update(f1_score.item(), n=batch_size)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    f1_score = pf1.result
    print("f1_score:",f1_score)
    print('* Pf1{pf1.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(pf1=metric_logger.pf1 ,top1=metric_logger.acc1, top2=metric_logger.acc2, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
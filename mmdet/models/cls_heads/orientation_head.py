# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses import Accuracy
from ..builder import HEADS, build_loss
# from ..utils import is_tracing
from .base_head import BaseHead


@HEADS.register_module()
class OrientationHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 roi_feat_size=7,
                 init_cfg=None,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 cal_acc=False):
        super(OrientationHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.cal_acc = cal_acc

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.roi_feat_size = roi_feat_size
        self.in_channels *= (self.roi_feat_size * self.roi_feat_size)

        print(self.in_channels)

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def simple_test(self, x):
        """Test without augmentation."""
        # if isinstance(x, tuple):
        #     x = x[-1]
        x = x.view(x.size(0), -1)
        cls_score = self.fc(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label):
        # if isinstance(x, tuple):
        #     x = x[-1]
        x = x.view(x.size(0), -1)
        cls_score = self.fc(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        # gt = F.one_hot(gt_label, num_classes=4)
        losses = self.loss(pred, gt_label)
        return losses

    def loss(self, cls_score, gt_label):
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score, gt_label)
        if self.cal_acc:
            # compute accuracy (not implemented)
            pass
        losses['loss'] = loss
        return losses

    def post_process(self, pred):
        # on_trace = is_tracing()
        # if torch.onnx.is_in_onnx_export() or on_trace:
        #     return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
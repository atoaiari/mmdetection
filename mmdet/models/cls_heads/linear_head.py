# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 roi_feat_size=7,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(LinearClsHead, self).__init__(*args, **kwargs)

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
        cls_score = self.fc(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label):
        # if isinstance(x, tuple):
        #     x = x[-1]
        x = x.view(x.size(0), -1)
        print(x.shape)
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label)
        return losses
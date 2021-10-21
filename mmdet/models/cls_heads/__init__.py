# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .linear_head import LinearClsHead
from .orientation_head import OrientationHead

__all__ = [
    'ClsHead', 'LinearClsHead', 'OrientationHead'
]
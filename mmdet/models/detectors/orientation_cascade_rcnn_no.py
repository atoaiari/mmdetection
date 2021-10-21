# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_head
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class OrientationCascadeRCNN(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 orientation_head=None):
        super(OrientationCascadeRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        assert orientation_head is not None
        self.orientation_head = build_head(orientation_head)


    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(OrientationCascadeRCNN, self).show_result(data, result, **kwargs)

    def forward_train(self,
                        img,
                        img_metas,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_ignore=None,
                        gt_masks=None,
                        gt_orientations=None,
                        proposals=None,
                        **kwargs):
            """
            Args:
                img (Tensor): of shape (N, C, H, W) encoding input images.
                    Typically these should be mean centered and std scaled.

                img_metas (list[dict]): list of image info dict where each dict
                    has: 'img_shape', 'scale_factor', 'flip', and may also contain
                    'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                    For details on the values of these keys see
                    `mmdet/datasets/pipelines/formatting.py:Collect`.

                gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                    shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

                gt_labels (list[Tensor]): class indices corresponding to each box

                gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                    boxes can be ignored when computing the loss.

                gt_masks (None | Tensor) : true segmentation masks for each box
                    used if the architecture supports a segmentation task.

                gt_orientations (list[Tensor]): orientation class indices corresponding to each box

                proposals : override rpn proposals with custom proposals. Use when
                    `with_rpn` is False.

            Returns:
                dict[str, Tensor]: a dictionary of loss components
            """
            x = self.extract_feat(img)

            losses = dict()

            # RPN forward and loss
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals

            roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                    gt_bboxes, gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            losses.update(roi_losses)

            return losses
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
from numpy.core.fromnumeric import argmax
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results
import numpy as np

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from sklearn.metrics import classification_report
from pycocotools.cocoeval import COCOeval



def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    orientation_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, orientation_result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)
        orientation_results.extend(orientation_result)

        for _ in range(batch_size):
            prog_bar.update()
    return results, orientation_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    # assert args.out or args.eval or args.format_only or args.show \
    #     or args.show_dir, \
    #     ('Please specify at least one operation (save/eval/format/show the '
    #      'results / save the results) with the argument "--out", "--eval"'
    #      ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    print(f"\n#### samples_per_gpu: {samples_per_gpu}\n")

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs, orientation_results = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)
        
        #####################################################################
        # orientation evaluation
        #####################################################################
        accepted_proposals = 0  # positive w.r.t to threshold on bbox score
        discarded_proposals = 0 # positive w.r.t to threshold on bbox score
        strong_proposals = 0    # if overlap with gt > gt_overlap_threshold
        weak_proposals = 0      # if overlap with gt < gt_overlap_threshold
        
        bbox_threshold = 0.5
        gt_overlap_threshold = 0.5

        bbox_results_with_gt = []

        cocoGt = dataset.coco
        imgIds = cocoGt.getImgIds()
        print(imgIds)

        pred_ori = []
        gt_ori = []
        mint_pred_ori = []
        mint_gt_ori = []

        for idx in range(len(dataset)):
            img_id = dataset.data_infos[idx]['id']
            gt_ann_ids = cocoGt.get_ann_ids(img_ids=[img_id])
            gt_ann_info = cocoGt.load_anns(gt_ann_ids)
            gt_ann = dataset._parse_ann_info(dataset.data_infos[idx], gt_ann_info)
            # img_id = dataset.img_ids[idx]
            # gt_ann_ids = cocoGt.getAnnIds(imgIds=[img_id])
            # gt_ann_info = cocoGt.loadAnns(gt_ann_ids)
            # gt_ann = dataset._parse_ann_info(gt_ann_info)
            
            gt_bboxes = np.array([gt_bbox for gt_bbox in gt_ann['bboxes']])

            # check_overlaps_with_gt = np.zeros(len(gt_bboxes))
            selected_pred = np.full(len(gt_bboxes), -1)
            selected_pred_max = np.zeros(len(gt_bboxes))

            det, _ = outputs[idx]
            ori = orientation_results[idx]

            # only one label (person)
            bboxes = det[0]
            print(f"\nimage {img_id} with {len([bb for bb in bboxes if bb[4]>=bbox_threshold])} valid proposals")
            for i in range(bboxes.shape[0]):
                bbox_score = float(bboxes[i][4])
                if bbox_score >= bbox_threshold:
                    accepted_proposals += 1
                    pred_bbox = np.array([bboxes[i][:4]])
                    overlaps = bbox_overlaps(pred_bbox, gt_bboxes)
                    max_idx = np.argmax(overlaps)

                    if np.max(overlaps) < gt_overlap_threshold:
                        weak_proposals += 1
                    else:
                        strong_proposals += 1

                    # check_overlaps_with_gt[max_idx] = 1
                    if np.max(overlaps) > selected_pred_max[max_idx]:
                        selected_pred[max_idx] = i
                        selected_pred_max[max_idx] = np.max(overlaps)
                    
                    gt_ori.append(int(np.argmax(gt_ann['orientations'][max_idx])))
                    pred_ori.append(int(np.argmax(ori[i]))) 

                    # for saving the detected bboxes as a new dataset for mebow testing
                    res_dict = dict()
                    res_dict['image_id'] = img_id
                    res_dict['bbox'] = xyxy2xywh(bboxes[i])
                    res_dict['score'] = bbox_score
                    res_dict['category_id'] = dataset.cat_ids[0]    # only person label
                    res_dict['orientation'] = int(np.argmax(ori[i]))
                    res_dict['orientation_score'] = float(np.max(ori[i]))
                    res_dict['gt_annotation_id'] = gt_ann_ids[max_idx]
                    gt_ann_original = cocoGt.loadAnns(gt_ann_ids[max_idx])
                    res_dict['gt_orientation'] = gt_ann_original[0]["orientation"]
                    bbox_results_with_gt.append(res_dict)
                else:
                    discarded_proposals += 1


            # if not check_overlaps_with_gt.all():
            #     print(f"GT bboxes not covered for img {img_id}")
            #     exit()

            for ix, sp in enumerate(selected_pred):
                if sp >= 0:
                    mint_pred_ori.append(int(np.argmax(ori[sp])))
                    mint_gt_ori.append(int(np.argmax(gt_ann["orientations"][ix])))
            
            # if not (selected_pred >= 0).all():
            #     print(f"!!!GT bboxes not covered for img {img_id}!!!")
            #     exit()
            # if len(set(selected_pred)) != len(gt_bboxes):
            #     print(f"Duplicated proposals")
            #     exit()

        if pred_ori:
            print("\nComplete classification report\n")
            print(classification_report(gt_ori, pred_ori))
            accc = calc_acc(gt_ori, pred_ori)
            accc_metrics = ["result", "excellent", "mid", "poor_225", "poor", "poor_45"]
            for accc_metric, accc_result in zip(accc_metrics, accc):
                print(f"{accc_metric}: {round(accc_result, 2)}")
        else:
            print("No positive bboxes")

        if mint_pred_ori:
            print("\nMint classification report\n")
            print(classification_report(mint_gt_ori, mint_pred_ori))

        print(f"accepted proposals: {accepted_proposals}")
        print(f"rejected proposals: {discarded_proposals}")
        print(f"weak proposals: {weak_proposals}")
        print(f"strong proposals: {strong_proposals}")

        # print(f"len(bbox_results_with_gt): {len(bbox_results_with_gt)}")
        # mmcv.dump(bbox_results_with_gt, osp.join(cfg.work_dir, f'bbox_results_with_gt_{int(bbox_threshold*10)}.json'))

def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]

def calc_acc(gt_ori, pred_ori):
    tot = len(pred_ori)
    result = 0
    excellent = 0
    mid = 0
    poor_225 = 0
    poor = 0
    poor_45 = 0
    for i in range(tot):
        diff = abs(pred_ori[i] - gt_ori[i]) * 5
        diff = min(diff, 360 - diff)
        result += diff
        if diff <= 45:
            poor_45 += 1
            if diff <= 30:
                poor += 1
                if diff <= 22.5:
                    poor_225 += 1
                    if diff <= 15:
                        mid += 1
                        if diff <= 5:
                            excellent += 1
    return [result/tot, excellent/tot, mid/tot, poor_225/tot, poor/tot, poor_45/tot]

if __name__ == '__main__':
    main()

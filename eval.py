import json
import os
import numpy as np

import PIL.Image as Image
from PIL import ImageDraw
import PIL
from merge import nms


def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """
    Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / (union + 1e-4)
    if exchange:
        ious = ious.T
    return ious


def tpfp_default(det_bboxes, gt_bboxes, gt_ignore, iou_thr, area_ranges=None):
    """
    Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): the detected bbox
        gt_bboxes (ndarray): ground truth bboxes of this image
        gt_ignore (ndarray): indicate if gts are ignored for evaluation or not
        iou_thr (float): the iou thresholds

    Returns:
        tuple: (tp, fp), two arrays whose elements are 0 and 1
    """
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fn = np.zeros((num_scales, num_gts), dtype=np.float32)
    # if there is no gt bboxes in this image, then all det bboxes within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + 1) * (det_bboxes[:, 3] - det_bboxes[:, 1] + 1)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp

    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore[matched_gt] or gt_area_ignore[matched_gt]):
                    if not fn[k, matched_gt]:
                        fn[k, matched_gt] = 1
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp, fn


def eval(draw_img=False):
    with open(r'exp\GTDetection\merge.json', 'r') as rf:
        data = json.load(rf)

    gts = []
    f = open(r'D:\Liang\Jilin\DXB\1460-1400\gt.txt', 'r')
    for line in f:
        line = line.split(',')
        line = [int(i) for i in line]
        gts.append([line[0], line[2], line[3], line[4], line[5]])
    f.close()

    num_tp, num_fp, num_fn = 0, 0, 0
    for file in os.listdir(r'D:\Liang\Jilin\DXB\1460-1400\img')[1:-1]:
        if draw_img:
            img = PIL.Image.open(os.path.join(r'D:\Liang\Jilin\DXB\1460-1400\img', file))
            draw = ImageDraw.ImageDraw(img)
        gt = [i[1:] for i in gts if i[0] == int(file[:-4])]
        pred = []
        objs = [i for i in data if i['image_id'] == int(file[:-4]) - 1]
        for obj in objs:
            if obj['score'] > 0.2:
                bbox = obj['bbox']
                bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
                bbox.append(obj['score'])
                pred.append(bbox)
        if len(pred) == 0:
            return 0, 0, 0
        retain = nms(np.array(pred), 0.5)
        pred_nms = []
        for i, obj in enumerate(pred):
            if i in retain:
                pred_nms.append(obj)
        tp, fp, fn = tpfp_default(np.array(pred_nms)[:, :-1], np.array(gt), np.zeros(len(gt)), 0.5)
        for i, obj in enumerate(pred_nms):
            if tp[0][i] == 1:
                if draw_img:
                    draw.rectangle(obj[:-1], fill=None, outline='blue', width=1)
                num_tp += 1
            else:
                if draw_img:
                    draw.rectangle(obj[:-1], fill=None, outline='red', width=1)
                num_fp += 1
        for i, obj in enumerate(gt):
            if fn[0][i] == 0:
                if draw_img:
                    draw.rectangle(obj, fill=None, outline='green', width=1)
                num_fn += 1
        if draw_img:
            img.save('results/%s.png' % file[:-4], quality=95)
            print('results/%s.png' % file[:-4])

    prec = num_tp / (num_tp + num_fp)
    rec = num_tp / (num_tp + num_fn)
    f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


if __name__ == '__main__':
    prec, rec, f1 = eval(draw_img=True)
    print('Prec: %.5f, Rec: %.5f, F1: %.5f' % (prec, rec, f1))

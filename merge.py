import json
import os

import numpy as np
import pycocotools.coco as coco
from cocoeval import COCOeval


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    retain = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        retain.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return retain


def merge(save_path):
    with open(os.path.join(save_path, 'results.json'), 'r') as rf:
        preds = json.load(rf)

    with open(r'E:\jqp\data\MOT\DXB\val.json', 'r') as rf:
        imgs = json.load(rf)['images']

    for obj in preds:
        img_name = [i['file_name'] for i in imgs if i['id'] == obj['image_id']][0]
        info = img_name[:-4].split('_')
        bbox = obj['bbox']
        bbox[0] = bbox[0] + int(info[3])
        bbox[1] = bbox[1] + int(info[4])
        obj['bbox'] = bbox
        obj['image_id'] = int(info[2])

    preds_nms = []
    for img_id in range(50):
        preds_img = []
        bboxes = np.array([i['bbox'] for i in preds if i['image_id'] == img_id + 1])
        scores = np.array([i['score'] for i in preds if i['image_id'] == img_id + 1])
        scores.shape = (1, scores.shape[0])
        bboxes = np.concatenate((bboxes, scores.T), axis=1)
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        retain = nms(bboxes, 0.5)
        for i in range(len(bboxes)):
            if i in retain:
                preds_img.append(preds[img_id * len(bboxes) + i - 1])
        # preds_img = sorted(preds_img, key=lambda x: x['score'], reverse=True)[:256]
        preds_nms += preds_img

    with open(os.path.join(save_path, 'merge.json'), 'w') as f:
        json.dump(preds_nms, f)


if __name__ == '__main__':
    merge(save_path=r'exp\WeightPseudoDetection_ResNet18_SD')
    tool = coco.COCO(r'D:\Liang\Jilin\DXB\annotation.json')
    coco_dets = tool.loadRes(os.path.join(r'exp\WeightPseudoDetection_ResNet18_SD', 'merge.json'))
    coco_eval = COCOeval(tool, coco_dets, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

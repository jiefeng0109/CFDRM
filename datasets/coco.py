from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import pycocotools.coco as coco
import torch.utils.data as data
from pycocotools.cocoeval import COCOeval
from trains.rank_CL import rank_bbox


class COCO(data.Dataset):
    num_classes = 1
    default_resolution = [512, 512]
    # default_resolution = [256, 256]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split, epoch=0):
        super(COCO, self).__init__()
        self.data_dir = r'E:\Jiang\data\MOT\DXB_train_val_test\pseudo_label'  # todo  E:\Jiang\data\MOT\xx
        # self.data_dir = r'E:\Jiang\data\RsCarData'

        if split == 'train':
            self.img_dir = os.path.join(self.data_dir, 'train')
            self.annot_path = os.path.join(self.data_dir, 'pseudo_%d.json' % epoch)  # update
            # 课程学习
            self.weight1, self.weight2 = rank_bbox(self.annot_path)
        else:
            self.img_dir = os.path.join(self.data_dir, 'val')
            self.annot_path = os.path.join(self.data_dir, 'gt.json')  # todo 正常训练
        # else:
        #     self.img_dir = os.path.join(self.data_dir, 'images/test1024/test/002')
        #     self.annot_path = os.path.join(self.data_dir, 'images/test1024/test/002/test.json')
        self.max_objs = 256  # '假设每张图片最大的目标个数不超过256'
        self.class_name = ['__background__', 'vehicle']
        self._valid_ids = [1]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt
        self.epoch = epoch

        print('==> initializing coco {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir, epoch):
        json.dump(self.convert_eval_format(results),
                  open('{}/results_%d.json'.format(save_dir) % (epoch-1), 'w'))

    def run_eval(self, results, save_dir, epoch):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir, epoch)
        # coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        # coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()

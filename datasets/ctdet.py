from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random

import cv2
import numpy as np
import torch.utils.data as data

from utils.image import color_aug
from utils.image import draw_dense_reg
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import get_affine_transform, affine_transform


class CTDetDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        # 突破口
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        # print(file_name)
        # print('###############')
        # print(len(anns))
        # print('################')
        # print(self.max_objs)  # 256
        # print('>>>>>>>>>>>>>>>>')
        num_objs = min(len(anns), self.max_objs)  # 可能会有问题，超过256就会存在一定问题

        img = np.load(img_path[:-4] + '.npy').astype(np.uint8)
        pre = img[:, :, 0:3]
        post = img[:, :, 6:9]
        img = img[:, :, 3:6]

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            # print('train')
            if not self.opt.not_rand_crop:
                # print("rand crop")
                # todo 为什么这里只做对img变化,下面的却要做对三张图像都进行变换
                # 长边缩放，随机裁剪，主要是随机变换中心点的位置，变换的范围是图像的四边向内移动border距离。
                # border的取值是如果图像尺寸超过256，border为128，否则为64
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                # print('scale and shift')
                # '缩放和平移，根据相应的因子确定新的中心点位置和长边'
                # '与裁剪的区别是是啥？
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                # #'水平翻转'
                # print('水平翻转')
                flipped = True
                img = img[:, ::-1, :]
                pre = pre[:, ::-1, :]
                post = post[:, ::-1, :]
                c[0] = width - c[0] - 1

        # '确定这些参数之后，对输入进行仿射变换
        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp_pre = cv2.warpAffine(pre, trans_input,
                                 (input_w, input_h),
                                 flags=cv2.INTER_LINEAR)
        inp_post = cv2.warpAffine(post, trans_input,
                                  (input_w, input_h),
                                  flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        inp_pre = (inp_pre.astype(np.float32) / 255.)
        inp_post = (inp_post.astype(np.float32) / 255.)

        # data augmentation
        if self.split == 'train' and not self.opt.no_color_aug:
            # print('color_aug')
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        inp = (inp - self.mean) / self.std
        inp_pre = (inp_pre - self.mean) / self.std
        inp_post = (inp_post - self.mean) / self.std

        inp = np.dstack((inp_pre, inp, inp_post))
        inp = inp.transpose(2, 0, 1)

        # '为输出做仿射变换做准备'
        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        # '准备数据集返回变量，主要是真值标签，如热力图、目标长度和宽度、偏移量'
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        hm_w = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        hm_pos = np.ones((num_classes, output_h, output_w), dtype=np.float32)
        hm_neg = np.ones((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.float32)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):  # 按实际有多少个目标来
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            # print(ann['weight'])
            weight = ann['weight']

            # weight = 1
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            # '输出的仿射变换，这些仿射变换由相应的函数完成，在实际编写中只需确定诸如中心点c、长边长度s等参数，便于在自己的数据集中使用'
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))  # '这个热力图的半径是根据目标的尺寸确定的'
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)  # 中点坐标
                ct_int = ct.astype(np.int32)  # 整型中点坐标

                draw_gaussian(hm[cls_id], ct_int, radius, 1)  # 更新高斯分布,绘制热力图，绘制在其所属类别的通道上
                # hm_w[k] = ann['weight']
                draw_gaussian(hm_w[cls_id], ct_int, radius, weight, 1)  # 更新高斯分布权重

                hm_pos[cls_id][int(min(bbox[0], 0)):int(max(bbox[2], output_w)), int(min(bbox[1], 0)):int(max(bbox[3], output_h))] = math.exp(weight)
                wh[k] = 1. * w, 1. * h  # 长宽 #'若object的个数不够最大数量，那么剩下的未填充的位置依然是0
                ind[k] = ct_int[1] * output_w + ct_int[0]  # ind 中心点的位置，用一维表示h*W+w
                reg[k] = ct - ct_int  # offset 由取整引起的误差
                reg_mask[k] = 1  # 有目标存在的位置，设为1
                # reg_mask[k] = weight  # mask权重
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]  # 类间长宽度不共享
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:  # False
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        # print(hm_w[np.where(hm_w!=0)])
        # print(hm[np.where(hm!=0)])
        # ret = {'input': inp, 'hm': hm, 'hm_w': hm_w,  'ind': ind, 'wh': wh}
        # neg
        # dataset = 'SkySat\\002\\'
        # if dataset == 'SkySat\\002\\':
        #     # data = np.loadtxt(r'datas\\MOT\\' + dataset +'pseudo_bbox.txt', delimiter=',',
        #     #                   dtype=int).tolist()  # todo
        #     neg_samples = np.loadtxt(r'datas\\MOT\\' + dataset + 'neg_samples.txt', delimiter=',',
        #                              dtype=int).tolist()
        # # print(file_name)
        # id = int(file_name.split('_')[2])
        # left = int(file_name.split('_')[3])
        # upper = int(file_name.split('_')[4][:-4])
        # right = left + 256
        # lower = upper + 256
        # # print(id,  left, upper)
        # neg_samples_pic = [i for i in neg_samples if i[0] == id]
        # for neg_sample in neg_samples_pic:
        #     if neg_sample[1] >= left and neg_sample[2] >= upper \
        #             and neg_sample[3] <= right and neg_sample[4] <= lower:
        #         # 找到与训练集对应的neg bboxes
        #         bbox = np.array(
        #             [neg_sample[1] - left, neg_sample[2] - upper, neg_sample[3] - left, neg_sample[4] - upper],
        #             dtype=np.float32)
        #         weight_neg = random.uniform(1, 2.7)
        #         if flipped:
        #             bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        #         # '输出的仿射变换，这些仿射变换由相应的函数完成，在实际编写中只需确定诸如中心点c、长边长度s等参数，便于在自己的数据集中使用'
        #         # 左上和右下进行仿射变换
        #         # print(bbox)
        #         # neg_feature = cv2.warpAffine(neg_multi, trans_output,(output_w, output_h),flags=cv2.INTER_LINEAR)
        #         # print(np.where(neg_feature!=1))
        #         bbox[:2] = affine_transform(bbox[:2], trans_output)
        #         bbox[2:] = affine_transform(bbox[2:], trans_output)
        #         bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
        #         bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
        #         h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        #         if h > 0 and w > 0:
        #             hm_neg[cls_id][int(min(bbox[0], 0)):int(max(bbox[2], output_w)),
        #             int(min(bbox[1], 0)):int(max(bbox[3], output_h))] = weight_neg
        ret = {'input': inp, 'hm': hm, 'hm_w': hm_w, 'hm_pos': hm_pos, 'hm_neg': hm_neg, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        # hm	热力图	[C,H,W]	float32
        # wh	目标的长度和宽度	[128,2]	float32
        # dense_wh	直接回归长宽度的map	[2,H,W]	float32
        # reg	下采样取整引起的偏移	[128,2]	float32
        # ind	中心点的位置(h*W+w),一维的表示方式	[128]	int64
        # reg_mask	固定长度的表示下是否存在关键点，最多128个目标	[128]	uint8
        # cat_spec_wh	长宽度预测类间不共享？没用到	[128,C*2]	float32
        # cat_spec_mask	？	[128,C*2]	uint8

        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret

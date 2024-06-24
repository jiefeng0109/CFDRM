import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
import json


def crop_img(load_path=r'E:\Jiang\data\MOT\\',
             save_path=r'E:\Jiang\data\MOT\SkySat',  # todo save_path
             # datasets=['1960-2690','250-2970'],
             # datasets=['1460-1400'],
             # datasets=['4570-3360', '8450-2180'],
             datasets=['SkySat\\001\\'],  # todo datasets
             cut_size=256,  # todo
             stride=256,
             padding=True
             ):
    save_id = 1
    save_label = []

    for dataset in datasets:
        file_list = os.listdir(os.path.join(load_path, dataset, 'img'))
        # data = np.loadtxt(os.path.join(load_path, dataset, 'gt.txt'), delimiter=',', dtype=int).tolist()
        # GMM生成的
        if dataset =='SkySat\\001\\':
            data = np.loadtxt(r'E:\Jiang\data\MOT\SkySat\001\gt\gt.txt', delimiter=',', dtype=int).tolist()
        elif dataset == 'SD\\9590-2960\\':
            data = np.loadtxt(r'E:\Jiang\data\MOT\SD\9590-2960\gt\gt.txt', delimiter=',', dtype=int).tolist()
        elif dataset == 'DXB\\1460-1400\\':
            data = np.loadtxt(r'E:\Jiang\data\MOT\DXB\1460-1400\gt\gt.txt', delimiter=',', dtype=int).tolist()
        # data = np.loadtxt(r'pseudo_bbox.txt', delimiter=',', dtype=int).tolist()  # todo np.loadtxt
        # print(data)
        for file in file_list[1:-1]:
            img_id = int(file[:-4])
            label = [i for i in data if i[0] == img_id]
            # print(label)

            img_pre = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id - 1)))
            img = Image.open(os.path.join(load_path, dataset, 'img', file))
            img_post = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id + 1)))
            img_w, img_h = img.size

            w = cut_size
            for h1 in range(0, img_h, w):
                for w1 in range(0, img_w, w):
                    print(w1, h1)
                    left = w1
                    upper = h1
                    # 最后不够时也直接切
                    if not padding:  # 没啥用忽然发现
                        right = min(w1 + w, img_w)  # 最后不够时也直接切
                        lower = min(h1 + w, img_h)
                    # 最后不够，重叠部分前面切
                    if padding:
                        print('回切')
                        # print(w)
                        right = min(w1 + w, img_w)
                        lower = min(h1 + w, img_h)
                        # print(right, lower)
                        # 宽不够了
                        if w1 + w > img_w and h1 + w <= img_h:
                            left = right - w
                        # 高不够了
                        if w1 + w <= img_w and h1 + w > img_h:
                            upper = lower - w
                        # 宽，高都不够了
                        if w1 + w > img_w and h1 + w > img_h:
                            upper = lower - w
                            left = right - w
                    for obj in label:
                        # 位于切割处的bbox会被舍弃，存在问题
                        if obj[2] >= left and obj[3] >= upper \
                                and obj[4] <= right and obj[5] <= lower:
                            # 相对crop的图片的位置下x,y
                            # w,h绝对长度
                            x, y = obj[2] - left, obj[3] - upper
                            w_qufen, h = obj[4] - obj[2], obj[5] - obj[3]
                            save_label.append([x, y, w_qufen, h, save_id])

                    print(left, upper, right, lower)
                    img_crop = img.crop([left, upper, right, lower])
                    img_pre_crop = img_pre.crop([left, upper, right, lower])
                    img_post_crop = img_post.crop([left, upper, right, lower])
                    # print(cut_size)
                    img_save = np.zeros((cut_size, cut_size, 9))
                    img_save[:, :, 0:3] = img_pre_crop
                    img_save[:, :, 3:6] = img_crop
                    img_save[:, :, 6:9] = img_post_crop
                    print(dataset)
                    np.save(os.path.join(save_path,
                                         'val\%03d_%s_%d_%d_%d.npy' % (save_id, dataset.replace(dataset.split('\\')[0]+'\\', '')[0:-1], img_id, left, upper)
                                         ),  # todo dataset[4:-1]
                            np.array(img_save, dtype=int))
                    save_id += 1
                    # save_id表示的切割之后的图片的id
                    # print(save_id)
            print(os.path.join(load_path, dataset, 'img', file))
        print(save_id, len(save_label))
    return save_label, save_id


def convert_json(data, save_path, img_size=256):  # todo
    coco = dict()
    coco['info'] = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []
    coco["licenses"] = []

    info = {"year": 2022.4,
            "version": '1.0',
            "description": 'val',
            "contributor": 'Jiang',
            "url": 'null',
            "date_created": '2022.4.23'
            }
    coco['info'] = info

    categories = {"id": 1,
                  "name": 'car',
                  "supercategory": 'null',
                  }
    coco['categories'].append(categories)

    for file in os.listdir(os.path.join(save_path, 'val')):  # todo
        # print(file)  # 182_9590-2960_46_0_500.npy
        # print(file.split('_'))
        # print(int(file.split('_')[0]))  # 取到第几张图片
        image = {"id": int(file.split('_')[0]),
                 "width": img_size,
                 "height": img_size,
                 "file_name": file,
                 "license": 0,
                 "flickr_url": 'null',
                 "coco_url": 'null',
                 "date_captured": '2022.4.11'
                 }
        coco['images'].append(image)

    for i, item in enumerate(data):
        print(len(data))
        print(i, item)
        # bbox[] is x,y,w,h
        print(item[4])
        bbox = [item[0], item[1], item[2], item[3]]
        # pseudo_point = [item[5], item[6]]
        annotation = {"id": i,
                      "image_id": item[4],
                      "category_id": 1,
                      "segmentation": [],
                      "area": item[2] * item[3],
                      "bbox": bbox,
                      "iscrowd": 0,
                      'ignore': 0,
                      # 'pseudo_point': pseudo_point,
                      # 'weight': Q[item[4] - 1]  # 理解为计算得到一个权重，然后加到了每一个
                      }
        coco['annotations'].append(annotation)
    print(len(coco['images']), len(coco['annotations']))

    with open(os.path.join(save_path, 'gt.json'), 'w') as f:  # todo
        json.dump(coco, f)

    return coco


if __name__ == "__main__":

    # save_label, img_num = crop_img()
    # coco_json = convert_json(data=save_label, save_path=r'E:\Jiang\data\MOT\SkySat')  # todo
    # 抉择选择哪个数据集  0：SkySat, 1:SD 2:DXB
    i = 2

    load_path = [
        r'E:\Jiang\data\MOT\\',
        r'E:\Jiang\data\MOT\\',
        r'E:\Jiang\data\MOT\\',
    ]
    save_path = [
        r'E:\Jiang\data\MOT\SkySat',
        r'E:\Jiang\data\MOT\SD',
        r'E:\Jiang\data\MOT\DXB',
    ]
    datasets = [
        ['SkySat\\001\\'],
        ['SD\\9590-2960\\'],
        ['DXB\\1460-1400\\'],
    ]
    cut_size = 256
    stride = 256
    padding = True

    save_label, _ = crop_img(
         load_path=load_path[i],
         save_path=save_path[i],  # todo
         datasets=datasets[i],  # todo
         cut_size=cut_size,  # todo   # jilin500  skysat 200
         stride=stride,
         padding=padding
    )
    save_path = [
        r'E:\Jiang\data\MOT\SkySat',
        r'E:\Jiang\data\MOT\SD',
        r'E:\Jiang\data\MOT\DXB'
    ]
    coco_json = convert_json(data=save_label, save_path=save_path[i])  # todo
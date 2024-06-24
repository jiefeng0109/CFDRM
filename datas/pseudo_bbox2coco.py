import numpy as np
from PIL import Image

import os
import json


def save_img(load_path='E:\\jqp\\data\\MOT\\SkySat',
             save_path='E:\\jqp\\data\\MOT\\SkySat',
             # datasets=['1960-2690','250-2970'],
             # datasets=['1460-1400'],
             # datasets=['4570-3360', '8450-2180'],
             datasets=['002']  # todo
             ):
    save_id = 1
    save_label = []

    for dataset in datasets:
        file_list = os.listdir(os.path.join(load_path, dataset, 'img'))
        # data = np.loadtxt(os.path.join(load_path, dataset, 'gt.txt'), delimiter=',', dtype=int).tolist()
        # GMM生成的
        # train
        data = np.loadtxt(r'pseudo_bbox.txt', delimiter=',', dtype=int).tolist()  # todo
        # val
        # data = np.loadtxt(r'E:\jqp\data\MOT\SkySat\001\gt\gt.txt', delimiter=',', dtype=int).tolist()
        # print(data)
        for file in file_list[1:-1]:  # 掐头去尾
            print(file)
            img_id = int(file[:-4])
            label = [i for i in data if i[0] == img_id]
            # print(label)

            img_pre = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id - 1)))
            img = Image.open(os.path.join(load_path, dataset, 'img', file))
            img_post = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id + 1)))
            print(img.size)
            img_w, img_h = img.size
            # print(img.size)
            for obj in label:
                x, y = obj[2], obj[3]
                w, h = obj[4] - obj[2], obj[5] - obj[3]
                save_label.append([x, y, w, h, save_id])
            img_save = np.zeros((img_h, img_w, 9))
            img_save[:, :, 0:3] = img_pre
            img_save[:, :, 3:6] = img
            img_save[:, :, 6:9] = img_post

            np.save(os.path.join(save_path, 'train\%03d_%s_%d.npy' % (save_id, dataset, img_id)), np.array(img_save, dtype=int))  # todo
            save_id += 1
            print(os.path.join(load_path, dataset, 'img', file))

    print(save_id, len(save_label))
    return save_label, save_id


def convert_json(data, save_path, img_w=400, img_h=600):  # img_h=600  # todo
    coco = dict()
    coco['info'] = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []
    coco["licenses"] = []

    info = {"year": 2022.4,
            "version": '1.0',
            "description": 'UnsupervisedDetection',
            "contributor": 'liang',
            "url": 'null',
            "date_created": '2022.4.23'
            }
    coco['info'] = info

    categories = {"id": 1,
                  "name": 'car',
                  "supercategory": 'null',
                  }
    coco['categories'].append(categories)

    for file in os.listdir(os.path.join(save_path, 'train')):  # todo
        print(file)  # 182_9590-2960_46_0_500.npy
        # print(file.split('_'))
        # print(int(file.split('_')[0]))  # 取到第几张图片
        image = {"id": int(file.split('_')[0]),
                 "width": img_w,
                 "height": img_h,
                 "file_name": file,
                 "license": 0,
                 "flickr_url": 'null',
                 "coco_url": 'null',
                 "date_captured": '2022.4.11'
                 }
        coco['images'].append(image)

    for i, item in enumerate(data):
        # bbox[] is x,y,w,h

        bbox = [item[0], item[1], item[2], item[3]]
        print(bbox)
        annotation = {"id": i,
                      "image_id": item[4],
                      "category_id": 1,
                      "segmentation": [],
                      "area": item[2] * item[3],
                      "bbox": bbox,
                      "iscrowd": 0,
                      'ignore': 0,
                      }
        coco['annotations'].append(annotation)
    print(len(coco['images']), len(coco['annotations']))

    with open(os.path.join(save_path, 'pseudo.json'), 'w') as f:   # todo
        json.dump(coco, f)

    return coco


if __name__ == "__main__":


    # datasets = ['DXB\\1460-1400\\', 'SD\\9590-2960\\', 'SkySat\\002\\']
    # # 先生成002再说
    # dataset = datasets[2]
    # load_path = 'E:\\jqp\\data\\MOT\\' + dataset + 'img\\'
    #
    # save_path = r'E:\jqp\data\MOT\SkySat'
    # annotations = []
    # f = open(r'E:\jqp\code\object_detection\UnsupervisedDetection\datas\pseudo_bbox.txt', 'r')

    save_label, img_num = save_img()
    coco_json = convert_json(data=save_label, save_path=r'E:\\jqp\\data\\MOT\\SkySat')

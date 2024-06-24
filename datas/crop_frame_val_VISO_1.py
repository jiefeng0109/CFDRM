import shutil

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
import json


def crop_img(load_path=r'E:\Jiang\data\RsCarData\images\\',
             save_path=r'E:\Jiang\data\RsCarData\images\test1024',  # todo
             datasets=['DXB\\1960-2690\\', 'DXB\\250-2970\\'],
             # datasets=['1460-1400'],
             # datasets=['SD\\4570-3360\\', 'SD\\8450-2180\\'],
             # datasets=['002', '001'],  # todo
             cut_size=512,  # todo   # jilin500  skysat 200
             stride=256,
             padding=True,
             ):
    save_id = 1
    save_label = []

    for dataset in datasets:
        print(dataset[9:-1])  # 002
        file_list = os.listdir(os.path.join(load_path, dataset, 'img1'))  # todo
        # print(os.path.join(load_path, dataset, 'img'))
        data = np.loadtxt(os.path.join(load_path, dataset, 'gt\gt.txt'), delimiter=',', dtype=int).tolist()
        # GMM生成的
        # data = np.loadtxt(r'E:\Jiang\data\new_VISO\mot\car\001\gt\gt.txt')
        # if dataset =='DXB\\1960-2690\\':
        #     data = np.loadtxt(r'E:\Jiang\data\MOT\DXB\1960-2690\gt\gt.txt', delimiter=',', dtype=int).tolist()  # todo
        # elif dataset =='DXB\\250-2970\\':
        #     data = np.loadtxt(r'E:\Jiang\data\MOT\DXB\250-2970\gt\gt.txt', delimiter=',', dtype=int).tolist()  # todo
        # data = np.loadtxt(r'E:\jqp\data\MOT\SkySat\001\gt\gt.txt', delimiter=',', dtype=int).tolist()
        # print(data)
        # 不取前后5帧测一次
        for file in file_list[4:-5]:  # todo
            # print(int(file_list[0][:-4]))
            # print(int(file_list[len(file_list)-1][:-4]))
            img_id = int(file[:-4])
            label = [i for i in data if i[0] == img_id]
            # print(label)
            # # 特殊情况
            # if img_id-5 <= 0:
            #     img_id = int(file_list[0][:-4])
            # elif img_id+5 >= int(file_list[len(file_list)+1][:-4]):
            #     img_id = int(file_list[len(file_list)+1][:-4])
            # img_pre1 = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id - 2)))
            img_pre = Image.open(os.path.join(load_path, dataset, 'img1', '%06d.jpg' % (max(img_id - 5, int(file_list[0][:-4])))))
            img = Image.open(os.path.join(load_path, dataset, 'img1', file))
            img_post = Image.open(os.path.join(load_path, dataset, 'img1', '%06d.jpg' % (min(img_id + 5, int(file_list[len(file_list)-1][:-4])))))
            # img_post1 = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id + 2)))

            img_w, img_h = img.size
            w = cut_size
            for h1 in range(0, img_h, w):
                for w1 in range(0, img_w, w):
                    print(w1, h1)
                    left = w1
                    upper = h1
                    # 最后不够时也直接切
                    if not padding:   # 没啥用忽然发现
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
                        # 左上右下
                        # viso需要转变一下
                        # print(obj)
                        # 注意循环利用的变量
                        obj_right = obj[2] + obj[4]
                        obj_lower = obj[3] + obj[5]

                        # 位于切割处的bbox会被舍弃，存在问题,应该像albu里面设定面积取消某些标注
                        if obj[2] >= left and obj[3] >= upper \
                                and obj_right <= right and obj_lower <= lower:
                            # 相对crop的图片的位置下x,y
                            # w,h绝对长度
                            # print(obj[2],obj[3],obj_right,obj_lower)
                            x, y = obj[2] - left, obj[3] - upper
                            w_qufen, h = obj_right - obj[2], obj_lower - obj[3]
                            save_label.append([x, y, w_qufen, h, save_id])

                            # print(save_label)

                    # print(left, upper, right, lower)
                    # todo
                    img_crop = img.crop([left, upper, right, lower])
                    img_pre_crop = img_pre.crop([left, upper, right, lower])
                    img_post_crop = img_post.crop([left, upper, right, lower])
                    # img_pre_crop1 = img_pre1.crop([left, upper, right, lower])
                    # img_post_crop1 = img_post1.crop([left, upper, right, lower])
                    # print(cut_size)
                    # todo
                    img_save = np.zeros((cut_size, cut_size, 9))
                    # img_save[:, :, 0:3] = img_pre_crop1
                    img_save[:, :, 0:3] = img_pre_crop
                    img_save[:, :, 3:6] = img_crop
                    img_save[:, :, 6:9] = img_post_crop
                    # img_save[:, :, 9:12] = img_post_crop1
                    # img_save.reshape(cut_size, cut_size, 3, 5)

                    np.save(os.path.join(save_path,
                                         r'test\\'+dataset[9:-1]+r'\\%03d_%s_%d_%d_%d.npy' % (save_id, dataset.replace(dataset.split('\\')[0]+'\\', '')[0:-1], img_id, left, upper)
                                         ),  # todo dataset[4:-1]
                            np.array(img_save, dtype=int))
                    save_id += 1
                    # save_id表示的切割之后的图片的id
                    # print(save_id)
            print(os.path.join(load_path, dataset, 'img1', file))

            # # 切的不是整图，但不一定有影响，后面改成整图切割
            # for i in range(img_w // stride):
            #     for j in range(img_h // stride):
            #         cut_x = stride * i
            #         cut_y = stride * j
            #         for obj in label:
            #             # 位于切割处的bbox会被舍弃，存在问题
            #             if obj[2] >= cut_x and obj[3] >= cut_y \
            #                     and obj[4] <= cut_x + cut_size and obj[5] <= cut_y + cut_size:
            #                 # 相对crop的图片的位置下x,y
            #                 # w,h绝对长度
            #                 x, y = obj[2] - cut_x, obj[3] - cut_y
            #                 w, h = obj[4] - obj[2], obj[5] - obj[3]
            #                 save_label.append([x, y, w, h, save_id])
            #                 # print(save_label)
            #         img_crop = img.crop([cut_x, cut_y, cut_x + cut_size, cut_y + cut_size])
            #         img_pre_crop = img_pre.crop([cut_x, cut_y, cut_x + cut_size, cut_y + cut_size])
            #         img_post_crop = img_post.crop([cut_x, cut_y, cut_x + cut_size, cut_y + cut_size])
            #
            #         img_save = np.zeros((cut_size, cut_size, 9))
            #         img_save[:, :, 0:3] = img_pre_crop
            #         img_save[:, :, 3:6] = img_crop
            #         img_save[:, :, 6:9] = img_post_crop
            #
            #         np.save(os.path.join(save_path,
            #                              'train\%03d_%s_%d_%d_%d.npy' % (save_id, dataset[4:-1], img_id, cut_x, cut_y)
            #                              ),  # todo dataset[4:-1]
            #                 np.array(img_save, dtype=int))
            #         save_id += 1
            #         # save_id表示的切割之后的图片的id
            #         # print(save_id)
            # print(os.path.join(load_path, dataset, 'img', file))
    print(save_id, len(save_label))
    return save_label, save_id


def convert_json(data, save_path, img_size=512):  # todo
    coco = dict()
    coco['info'] = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []
    coco["licenses"] = []

    info = {"year": 2022.6,
            "version": '1.0',
            "description": 'supervisedDetection',
            "contributor": 'Jiang',
            "url": 'null',
            "date_created": '2022.6.13'
            }
    coco['info'] = info

    categories = {"id": 1,
                  "name": 'car',
                  "supercategory": 'null',
                  }
    coco['categories'].append(categories)

    for file in os.listdir(save_path):  # todo
        # if '001' in file:
        #     w, h = 400, 400
        # else:
        #     w, h = 400, 600
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
                 "date_captured": '2022.6.13'
                 }
        coco['images'].append(image)

    for i, item in enumerate(data):
        # bbox[] is x,y,w,h
        bbox = [item[0], item[1], item[2], item[3]]
        annotation = {"id": i,
                      "image_id": item[4],
                      "category_id": 1,
                      "segmentation": [],
                      "area": item[2] * item[3],
                      "bbox": bbox,
                      "iscrowd": 0,
                      'ignore': 0,
                      # 'weight': Q[item[4] - 1]  # 理解为计算得到一个权重，然后加到了每一个
                      }
        coco['annotations'].append(annotation)
    print(len(coco['images']), len(coco['annotations']))

    with open(os.path.join(save_path, 'test.json'), 'w') as f:  # todo
        json.dump(coco, f)

    return coco


def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath, ignore_errors=True)
        os.mkdir(filepath)


if __name__ == "__main__":
    nums = [2, 3, 5, 6, 8, 9, 10]
    datasets = ['test1024\\%03d\\' %(i) for i in nums]
    print(datasets)
    for dataset in datasets:
        print(dataset)
        list_dataset = []
        list_dataset.append(dataset)
        setDir(r'E:\Jiang\data\RsCarData\images\test1024\test\\'+dataset[9:-1])
        save_label, img_num = crop_img(datasets=list_dataset)
        coco_json = convert_json(data=save_label, save_path=r'E:\Jiang\data\RsCarData\images\test1024\test\\'+dataset[9:-1])  # todo
        list_dataset.clear()


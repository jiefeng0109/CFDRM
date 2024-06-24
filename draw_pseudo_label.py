import os
import numpy as np
import cv2

import PIL.Image as Image
from PIL import ImageDraw, ImageFont
import json
import PIL
from merge import nms

# with open(r'exp\GTDetection\merge.json', 'r') as rf:
#     preds = json.load(rf)

# with open(r'D:\Liang\Jilin\DXB\pseudo_1.json', 'r') as rf:
#     dets = json.load(rf)['annotations']
# choose
dataset = 0
# todo
load_dets_path = [
    r'E:\Jiang\data\MOT\SkySat\pseudo_label\pseudo_25.json',
    r'E:\Jiang\data\MOT\SD\pseudo_label\pseudo_1.json',
    r'E:\Jiang\data\MOT\DXB\pseudo_label\pseudo_10.json',
    r'E:\Jiang\data\RsCarData\pseudo_0.json',
]

load_gts_path = [
    r'E:\Jiang\data\MOT\SkySat\bbox_centernet\pseudo.json',
    r'E:\Jiang\data\MOT\SD\bbox_centernet\pseudo.json',
    r'E:\Jiang\data\MOT\DXB\bbox_centernet\pseudo.json',
    r'E:\Jiang\data\RsCarData\bbox_centernet\train_5.json'
]

val_path = [
    r'E:\Jiang\data\MOT\SkySat\pseudo_label\train',
    r'E:\Jiang\data\MOT\SD\pseudo_label\train',
    r'E:\Jiang\data\MOT\DXB\pseudo_label\train',
    r'E:\Jiang\data\RsCarData\train',
]
# todo
saveimg_path = [
    'src/SkySat/train_25/%s.png',
    'src/SD/train_1/%s.png',
    'src/DXB/train_10/%s.png',
    'src/VISO/train_GMM_0/%s.png',
]

with open(load_dets_path[dataset], 'r') as rf:  # todo
    dets = json.load(rf)['annotations']

with open(load_gts_path[dataset], 'r') as rf:  # todo E:\Jiang\data\MOT\SD\bbox_centernet\gt.json
    gts = json.load(rf)['annotations']
# print(type(json.load(rf)))

max_n = 0
for file in os.listdir(val_path[dataset]):  # todo E:\Jiang\data\MOT\SD\bbox_centernet\val
    # img = PIL.Image.open(os.path.join(r'D:\Liang\Jilin\SD\9590-2960\img', file))
    img = np.load(os.path.join(val_path[dataset], file))[:, :, 3:6]  # todo  # 注意15通道还是9通道
    img_id = int(file.split('_')[0])  # todo
    print(img_id)
    img = Image.fromarray(np.uint8(img))
    draw = ImageDraw.ImageDraw(img)
    font = ImageFont.truetype('simhei.ttf', 12)
    # print(dets)

    objs = [i for i in dets if i['image_id'] == img_id]
    for i, obj in enumerate(objs):
        # if obj['score'] > 0.2:
        bbox = obj['bbox']
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        # bbox[0] = bbox[0] - 1.5
        # bbox[1] = bbox[1] - 1.5
        # bbox[2] = bbox[2] + 1.5
        # bbox[3] = bbox[3] + 1.5
        # draw.rectangle(bbox, fill=None, outline='blue', width=1)  # 伪标签框
        # draw.text((bbox[2], bbox[1]), '%.2f' % obj['score'], fill=(0, 255, 0), font=font)
        # point = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
        # draw.point(point, 'red')

    objs = [i for i in gts if i['image_id'] == img_id]
    count = 0
    for i, obj in enumerate(objs):
        bbox = obj['bbox']
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        bbox[0] = bbox[0] - 1.5
        bbox[1] = bbox[1] - 1.5
        bbox[2] = bbox[2] + 1.5
        bbox[3] = bbox[3] + 1.5
        print(bbox)
        if bbox[3] - bbox[1] > 20:

            bbox[1] = bbox[1] + 10
            bbox[3] = bbox[3] - 10
            bbox[0] = bbox[0] + 3
            bbox[2] = bbox[2] - 3
            count = count+1

        # draw.rectangle(bbox, fill=None, outline='blue', width=1)  # gt
    print(count)
    # objs = [i for i in preds if i['image_id'] == img_id]
    # for i, obj in enumerate(objs):
    #     if obj['score']>0.05:
    #         bbox = obj['bbox']
    #         bbox[2] = bbox[0] + bbox[2]
    #         bbox[3] = bbox[1] + bbox[3]
    #         draw.rectangle(bbox, fill=None, outline='red', width=1)

    img.save(saveimg_path[dataset] % file[:-4], quality=95)   # todo
    print(saveimg_path[dataset] % file[:-4])  # todo


# for i, file in enumerate(os.listdir(r'D:\Liang\Jilin\SD\vehicle')):
#     img = np.load(os.path.join(r'D:\Liang\Jilin\SD\vehicle', file)).astype(np.uint8)[:, :, 0:3]
#     cv2.imwrite(os.path.join(r'viso', file[:-4] + '_0.jpg'), img)
#
#     img = np.load(os.path.join(r'D:\Liang\Jilin\SD\vehicle', file)).astype(np.uint8)[:, :, 3:6]
#     cv2.imwrite(os.path.join(r'viso', file[:-4] + '_1.jpg'), img)
#
#     img = np.load(os.path.join(r'D:\Liang\Jilin\SD\vehicle', file)).astype(np.uint8)[:, :, 6:9]
#     cv2.imwrite(os.path.join(r'viso', file[:-4] + '_2.jpg'), img)
#
#     print(os.path.join('viso', file[:-4] + '.jpg'))
#     if i == 20:
#         break

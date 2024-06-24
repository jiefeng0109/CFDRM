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

def iou(a, b):
    x1 = a[0]
    x2 = b[0]
    y1 = a[1]
    y2 = b[1]
    X1 = a[2]
    X2 = b[2]
    Y1 = a[3]
    Y2 = b[3]
    xx = max(x1, x2)
    XX = min(X1, X2)
    yy = max(y1, y2)
    YY = min(Y1, Y2)
    m = max(0., XX - xx)
    n = max(0., YY - yy)
    Jiao = m * n
    Bing = (X1 - x1) * (Y1 - y1) + (X2 - x2) * (Y2 - y2) - Jiao
    return Jiao / Bing



dataset = 3

load_dets_path = [
    'exp/SkySat/results_17.json',
    'exp/SD/results_25.json',
    'exp/DXB/results_22.json',
    'exp/VISO/results_0.json',
]

load_gts_path = [
    'E:\Jiang\data\MOT\SkySat\pseudo_label\gt.json',
    'E:\Jiang\data\MOT\SD\pseudo_label\gt.json',
    'E:\Jiang\data\MOT\DXB\pseudo_label\gt.json',
    'E:\Jiang\data\RsCarData\gt.json'
]

val_path = [
    'E:\Jiang\data\MOT\SkySat\pseudo_label\\val',
    'E:\Jiang\data\MOT\SD\pseudo_label\\val',
    'E:\Jiang\data\MOT\DXB\pseudo_label\\val',
    'E:\Jiang\data\RsCarData\\val',
]

saveimg_path = [
    'src/SkySat/val/%s.png',
    'src/SD/val/%s.png',
    'src/DXB/val/%s.png',
    'src/VISO/val/%s.png',
]

with open(load_dets_path[dataset], 'r') as rf:  # todo
    dets = json.load(rf)

with open(load_gts_path[dataset], 'r') as rf:  # todo E:\Jiang\data\MOT\SD\bbox_centernet\gt.json
    gts = json.load(rf)['annotations']
# print(type(json.load(rf)))
right = wrong = miss = 0


max_n = 0
for file in os.listdir(val_path[dataset]):  # todo E:\Jiang\data\MOT\SD\bbox_centernet\val
    # img = PIL.Image.open(os.path.join(r'D:\Liang\Jilin\SD\9590-2960\img', file))
    img = np.load(os.path.join(val_path[dataset], file))[:, :, 3:6]  # todo  # 注意15通道还是9通道
    img_id = int(file.split('_')[0])
    img = Image.fromarray(np.uint8(img))
    draw = ImageDraw.ImageDraw(img)
    font = ImageFont.truetype('simhei.ttf', 12)

    detections = [i for i in dets if i['image_id'] == img_id]
    labels = [i for i in gts if i['image_id'] == img_id]
    # 每张图片的
    get = np.zeros(2000)  # 图片大小，索引每个左上角位置
    for i, det in enumerate(detections):
        if det['score'] > 0.15:
            box1 = det['bbox']
            box1[0] -= 1
            box1[1] -= 1
            box1[2] += 1
            box1[3] += 1
            # print(box1)
            box3 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
            box1[2] = box1[0] + box1[2]
            box1[3] = box1[1] + box1[3]
            # print(box1)
            # draw.rectangle(box1, fill=None, outline='blue', width=1)

            flag = False
            for car in labels:
                # print(car)
                box2 = car['bbox']
                # print(box2)
                box4 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
                # box2[2] = box2[0] + box2[2]
                # box2[3] = box2[1] + box2[3]
                # print(box1, box2)
                if iou(box3, box4) >= 0.3:
                    # print('true')
                    draw.rectangle(box1, fill=None, outline='blue', width=1)
                    # print(box1, box2)
                    # frame = cv2.rectangle(frame,
                    #                       (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (255, 0, 0), 1)
                    # frame = cv2.rectangle(frame, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])),
                    #                       (255, 0, 0), 1)  # 蓝
                    flag = True
                    right += 1
                    get[box2[0]] = 1
                    break
            # 统计错误检测
            if not flag:
                if box1[1] <= 5 or box1[3] >= 250:
                    print('true')
                    continue
                else:
                    draw.rectangle(box1, fill=None, outline='red', width=1)
                wrong += 1
                # frame = cv2.rectangle(frame,
                #                       (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (0, 0, 255), 1)  # 红
            # 统计漏检
    for car in labels:
        box2 = car['bbox']
        box2[2] = box2[0] + box2[2]
        box2[3] = box2[1] + box2[3]
        if get[box2[0]] == 0:
            draw.rectangle(box2, fill=None, outline='yellow', width=1)
            miss += 1
            # frame = cv2.rectangle(frame,
            #                       (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])), (0, 255, 255), 1)  # 黄


    # print(len(detections))
    # print(len(labels))
    # get = np.zeros(2000)
    # objs = [i for i in dets if i['image_id'] == img_id]
    # for i, obj in enumerate(objs):
    #     if obj['score'] > 0.2:
    #         bbox = obj['bbox']
    #         print(bbox)
    #         bbox[2] = bbox[0] + bbox[2]
    #         bbox[3] = bbox[1] + bbox[3]
    #         draw.rectangle(bbox, fill=None, outline='blue', width=1)
            # draw.text((bbox[2], bbox[1]), '%.2f' % obj['score'], fill=(0, 255, 0), font=font)
    #
    # objs = [i for i in gts if i['image_id'] == img_id]
    # for i, obj in enumerate(objs):
    #     bbox = obj['bbox']
    #     bbox[2] = bbox[0] + bbox[2]
    #     bbox[3] = bbox[1] + bbox[3]
    #     draw.rectangle(bbox, fill=None, outline='red', width=1)



    #     for car in gt:
    #         box2 = [car[2], car[3], car[2]+car[4], car[3]+car[5]]
    #         # 统计检测正确
    #         if IOU(box1, box2) >= 0.1:
    #             frame = cv2.rectangle(frame, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])),
    #                                   (255, 0, 0), 1)
    #             # frame = cv2.rectangle(frame, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])),
    #             #                       (255, 0, 0), 1)  # 蓝
    #             flag = 1
    #             right += 1
    #             get[car[2]] = 1
    #             break
    #     # 统计错误检测
    #     if flag == 0:
    #         frame = cv2.rectangle(frame, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (0, 0, 255),
    #                               1)  # 红
    #         wrong += 1
    # # 统计漏检
    # for car in gt:
    #     if get[car[2]] == 0:
    #         box2 = [car[2], car[3], car[2]+car[4], car[3]+car[5]]
    #         frame = cv2.rectangle(frame, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])), (0, 255, 255), 1)  # 黄
    #         miss += 1

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

print(right, wrong, miss)
prec = right / (right + wrong)
rec = right / (right + miss)
f1 = 2 * prec * rec / (prec + rec)
# t = np.array(t)
print(prec * 100, rec * 100, f1 * 100)

import os
import numpy as np
import json
from progress.bar import Bar


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


def bbox_iou_single(box1, box2):
    '''
    box:[top, left, bottom, right]
    '''
    inter_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if inter_h < 0 or inter_w < 0 else inter_h * inter_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return inter / union


def update_annotation(dets_file, ann_file, img_path, save_path, epoch):
    '''
        pred:
        1.判断点在框里面
        2.判断score大于此前更新后的框的score
        3.判断iou>0.7 正框
        4.如果都符合更新，否则不更新

        origin_point: 永远不变（还是点也可以不断迭代更新到中心点）
        origin_bbox: 初始score为0.3（之后可以通过一个计算来实现，比如点和框中心点的距离，越近score越大，越远score越小）


    '''
    with open(dets_file, 'r') as rf:
        preds = json.load(rf)
        #  "image_id": int(image_id),
        #  "category_id": int(category_id),
        #  "bbox": bbox_out,
        #  "score": float("{:.2f}".format(score))

    with open(ann_file, 'r') as rf:
        coco = json.load(rf)
    # # # pseudo label point
    # with open(point_file, 'r') as rf:
    #     point = np.loadtxt(rf, delimiter=',', dtype=int).tolist()
    print(len(coco["annotations"]))
    img_width = coco['images'][0]['width']
    img_height = coco['images'][0]['height']

    img_names = os.listdir(img_path)
    update_coco = []
    obj_id = 0
    bar = Bar('{}'.format('updating'), max=len(img_names))
    for i, img_name in enumerate(img_names):
        img_id = int(img_name.split('_')[0])  # 1,2,3
        # print(img_id)
        update_frame = []
        weight = []
        img = np.load(os.path.join(img_path, img_name)).astype(np.uint8)  # todo why int
        objs = [obj for obj in preds if obj['image_id'] == img_id]
        pseudo_gts = [pseudo_gt for pseudo_gt in coco['annotations'] if pseudo_gt['image_id'] == img_id]
        # print(pseudo_gts)
        # pseudo_points = [pseudo_point for pseudo_point in point if pseudo_point[0] == img_id]
        # print(len(objs))
        # print(len(pseudo_gts))
        # print(pseudo_gts)
        # print(len(pseudo_points))
        # for obj in objs:
        #     # pred
        #     # bbox = [int(obj) for obj in obj['bbox']]  # todo int influence
        #     bbox = [int(car) for car in obj['bbox']]  # todo int influence
        #     bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1,y1,x2,y2
        #     # avoid out
        #     bbox[0], bbox[1] = max(bbox[0], 0), max(bbox[1], 0)
        #     bbox[2], bbox[3] = min(bbox[2], img_width), min(bbox[3], img_height)  # 取整可能出界？
        #     # print(pseudo_gts)
        #     for annotation in pseudo_gts:
        #         # print(annotation)
        #         # gt
        #         car = annotation['bbox']
        #         box2 = [car[0], car[1], car[0] + car[2], car[1]+car[3]]
        #         point = annotation['pseudo_point']
        #         # print(annotation)
        #         # 统计检测正确
        #         # 仅保留那些包含点的预测框
        #         # for m in range(len(pseudo_gts)):
        #         #     # print(m)
        #         #     # print(obj['score'])
        #         #     # and bbox_iou_single(bbox, box2) >= 0.3\
        #         #     # print(bbox)
        #         if bbox[0] < point[0] < bbox[2] and bbox[1] < point[1] < bbox[3] \
        #                 and obj['score'] >= 0.1:
        #                     print('update')
        #                     print(annotation)
        #                     # update
        #
        #                     annotation.update({
        #                         "area": (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]),
        #                         "bbox": [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
        #                         "weight": obj['score']
        #                     })
        #                     # print(obj['bbox'])
        #                     # print(annotation)

        for annotation in pseudo_gts:
            # print(annotation)
            # gt
            car = annotation['bbox']
            bbox2 = [car[0], car[1], car[0] + car[2], car[1]+car[3]]
            point = annotation['pseudo_point']
            # print(annotation)
            for obj in objs:
                # pred
                # bbox = [int(obj) for obj in obj['bbox']]  # todo int influence
                bbox = [int(car) for car in obj['bbox']]  # todo int influence
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1,y1,x2,y2
                # avoid out
                bbox[0], bbox[1] = max(bbox[0], 0), max(bbox[1], 0)
                bbox[2], bbox[3] = min(bbox[2], img_width), min(bbox[3], img_height)  # 取整可能出界？
                # print(pseudo_gts)
                # 统计检测正确
                # 仅保留那些包含点的预测框
                # for m in range(len(pseudo_gts)):
                #     # print(m)
                #     # print(obj['score'])
                #     # and bbox_iou_single(bbox, box2) >= 0.3\
                #     # print(bbox)
                # 固定值0.1
                if bbox[0] < point[0] < bbox[2] and bbox[1] < point[1] < bbox[3] \
                        and obj['score'] >= annotation['weight']:
                    print('update')

                    # print(annotation)
                    # update
                    annotation.update({
                        "area": (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]),
                        "bbox": [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
                        "weight": obj['score']
                    })
        # print(i)
            # print(type(annotation))
            update_coco.append(annotation)

            # print(len(update_coco))

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} ' \
            .format(i, len(img_names), total=bar.elapsed_td, eta=bar.eta_td)

        bar.next()

    # coco['annotations'] = update

    bar.finish()
    # print(len(update_coco))
    convert_json(update_coco, save_path, img_size=256, epoch=epoch)
    # update
    # with open(ann_file, 'w') as f:

    #     json.dump(coco, f)

    # with open(ann_file[:-5] + '_%d.json' % epoch, 'w') as f:
    #     json.dump(coco, f)

    return


def convert_json(data, save_path, epoch, img_size=256):  # todo
    coco = dict()
    coco['info'] = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = data  # todo
    coco['categories'] = []
    coco["licenses"] = []

    info = {"year": 2022.9,
            "version": '1.0',
            "description": 'UnsupervisedDetection',
            "contributor": 'Jiang',
            "url": 'null',
            "date_created": '2022.9.03'
            }
    coco['info'] = info

    categories = {"id": 1,
                  "name": 'car',
                  "supercategory": 'null',
                  }
    coco['categories'].append(categories)

    for file in os.listdir(os.path.join(save_path, 'train')):  # todo
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
                 "date_captured": '2022.4.11'
                 }
        coco['images'].append(image)

    # for i, item in enumerate(data):
    #     # bbox[] is x,y,w,h
    #     bbox = [item[0], item[1], item[2], item[3]]
    #     pseudo_point = [item[5], item[6]]
    #     annotation = {"id": i,
    #                   "image_id": item[4],
    #                   "category_id": 1,
    #                   "segmentation": [],
    #                   "area": item[2] * item[3],
    #                   "bbox": bbox,
    #                   "iscrowd": 0,
    #                   'ignore': 0,
    #                   'pseudo_point': pseudo_point,
    #                   'weight': 0.3  # 理解为计算得到一个权重，然后加到了每一个
    #                   }
    # coco['annotations'].append(data)
    print(len(coco['images']), len(coco['annotations']))

    # with open(os.path.join(save_path, 'pseudo.json'), 'w') as f:  # todo
    #     json.dump(coco, f)
    with open(os.path.join(save_path, 'pseudo_%d.json' % epoch), 'w') as f:
        json.dump(coco, f)
    return coco


if __name__ == '__main__':
    # opt = opts().parse()
    # opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # resnet = resnet18(num_classes=2)
    # resnet.load_state_dict(torch.load('../resnet.pkl'))
    # resnet.to(device=opt.device).eval()

    # print('resnet loaded!')

    update_annotation(r'..\\exp\SD\results.json',
                      r'E:\Jiang\data\MOT\SD\pseudo_label\pseudo_0.json',
                      r'E:\Jiang\data\MOT\SD\pseudo_label\train',  # 这应该有两个文件
                      r'E:\Jiang\data\MOT\SD\pseudo_label',
                      1)
    # update_annotation('D:/Liang/UnsupervisedDetection/exp/WeightPseudoDetection_ResNet18_SD/results.json',
    #                   'D:/Liang/Jilin/SD/pseudo.json',
    #                   'D:/Liang/Jilin/SD/val',
    #                   resnet,
    #                   0,
    #                   device=opt.device)

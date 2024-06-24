import os
import numpy as np
import json
from progress.bar import Bar


def cal_weight(dets_file, ann_file, img_path, save_path, epoch):
    with open(ann_file, 'r') as rf:
        coco = json.load(rf)
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
        # objs = [obj for obj in preds if obj['image_id'] == img_id]
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
            bbox2 = [car[0], car[1], car[0] + car[2], car[1] + car[3]]
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
                        "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                        "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
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
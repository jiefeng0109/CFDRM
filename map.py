import numpy as np
import json
import matplotlib.pyplot as plt


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# 主函数，读取预测和真实数据，计算Recall, Precision, AP
def voc_eval(detpath,
             annopath,
             ovthresh=1,
             use_07_metric=False):
    """
    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections detpath.format(classname) 需要计算的类别的txt文件路径.
    annopath: Path to annotations annopath.format(imagename) label的xml文件所在的路径
    imagesetfile: 测试txt文件，里面是每个测试图片的地址，每行一个地址
    classname: 需要计算的类别

    [ovthresh]: IOU重叠度 (default = 0.5)
    [use_07_metric]: 是否使用VOC07的11点AP计算(default False)
    """

    with open(annopath, 'r') as rf:
        annotations = json.load(rf)  # gt

    with open(detpath, 'r') as rf:
        dets = json.load(rf)
        # print(dets)
    # print(annotations)
    # read list of image
    imagenames = [i['file_name'] for i in annotations['images']]
    # print(imagenames)
    # 对每张图片的xml获取函数指定类的bbox等
    class_recs = {}
    npos = 0  # npos标记的目标数量
    # print('-----------------------')
    for img_id, imagename in enumerate(imagenames):
        # print('-----------')
        # print(img_id+1)
        # print(imagename)
        R = [obj for obj in annotations['annotations'] if obj['image_id'] == img_id + 1]
        # print(R)
        # print(len(R))
        bbox = np.array([x['bbox'] for x in R])
        # print(bbox)
        # 存在切块一整张不具有标注的情况
        if len(bbox) == 0:
            continue
        else:
            bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
            bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
            det = [False] * len(R)  # 每一个目标框对应一个det[i]，用来判断该目标框是否已经处理过
            npos = npos + len(R)  # 计算总的目标个数
            class_recs[img_id + 1] = {'bbox': bbox, 'det': det}
            # img_id+1  100   imagename  1000_003_254_512_512.npy
    # print(class_recs[402])
    # print(npos)
    splitlines = []
    for obj in dets:
        if obj['score'] > 0.3:  # 是否过滤检测框，默认为false
            # 检测的框
            # < image id > < confidence > < left > < top > < right > < bottom >
            image_id = obj['image_id']
            confidence = obj['score']
            left, top = obj['bbox'][0], obj['bbox'][1]
            right, bottom = obj['bbox'][0] + obj['bbox'][2], obj['bbox'][1] + obj['bbox'][3]
            splitlines.append([image_id, confidence, left, top, right, bottom])
    # print(len(splitlines))
    image_ids = [x[0] for x in splitlines]  # 图片ID
    confidence = np.array([x[1] for x in splitlines])  # 置信度得分
    BB = np.array([[z for z in x[2:]] for x in splitlines])  # bounding box数值,这是按0,1,2走的

    # 对confidence的index根据值大小进行降序排列。
    sorted_ind = np.argsort(-confidence)
    # print(sorted_ind[0:10])
    BB = BB[sorted_ind, :]  # 重排bbox，由大概率到小概率。
    # print(BB[0:10])
    image_ids = [image_ids[x] for x in sorted_ind]  # 图片重排，由大概率到小概率。
    # print(image_ids[0:10])
    # print(len(image_ids))
    # print(splitlines[54000])
    # print(image_ids[0])
    # print(class_recs[image_ids[0]])
    # go down dets and mark TPs and FPs
    nd = len(image_ids)  # 预测了126400个框
    print('预测框个数')
    print(nd)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    # 对于每一个预测框，先找到它所在图片的每一个gt,然后计算每一个预测框与gt之间的iou，大于iou阈值且没有与其他框匹配则为TP，否则为FP
    for d in range(nd):  # 遍历每一个框
        # print(d)  # 如果存在切割图片没有标签的，还是会报错
        if image_ids[d] not in class_recs:
            fp[d] = 1.
            continue   # 这里报错keyerror是因为对应的真实框可能无目标，但是检测框却认为有目标，说明这是个fp
        else:
            R = class_recs[image_ids[d]]  # 得到图像名字为image_ids[d]真实的目标框信息
            # print(R)
            bb = BB[d, :].astype(float)  # 得到图像名字为image_ids[d]检测的目标框坐标
            # print(bb)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)  # 取出图像名字为image_ids[d]真实的目标框坐标
            # print(BBGT)

        if BBGT.size > 0:
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            # print(BBGT[:, 0])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                   (BBGT[:, 2] - BBGT[:, 0]) *
                   (BBGT[:, 3] - BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)  # 检测到的目标框可能预若干个真实目标框都有交集，选择其中交集最大的
            jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            if not R['det'][jmax]:  # 该真实目标框是否已经统计过
                tp[d] = 1.  # 将tp对应第d个位置变成1
                R['det'][jmax] = 1  # 将该真实目标框做标记
            else:
                fp[d] = 1.   # 虽然该框大于某一个阈值，但是这个gt框已与其他框匹配，所以该预测框为FP
        else:
            fp[d] = 1.  # 该预测框与每一个真实框都小于阈值，所以没有匹配到

    # for d in image_ids:
    #     R = class_recs[d]
    #     # print(R)
    #     bb = BB[image_ids.index(d), :].astype(float)
    #     ovmax = -np.inf
    #     BBGT = R['bbox'].astype(float)
    #
    #     if BBGT.size > 0:
    #         ixmin = np.maximum(BBGT[:, 0], bb[0])
    #         iymin = np.maximum(BBGT[:, 1], bb[1])
    #         ixmax = np.minimum(BBGT[:, 2], bb[2])
    #         iymax = np.minimum(BBGT[:, 3], bb[3])
    #         iw = np.maximum(ixmax - ixmin + 1., 0.)
    #         ih = np.maximum(iymax - iymin + 1., 0.)
    #         inters = iw * ih
    #         uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
    #                (BBGT[:, 2] - BBGT[:, 0] + 1.) *
    #                (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
    #         overlaps = inters / uni
    #         ovmax = np.max(overlaps)
    #         jmax = np.argmax(overlaps)
    #     if ovmax > ovthresh:
    #         if not R['det'][jmax]:
    #             tp[image_ids.index(d)] = 1.
    #             R['det'][jmax] = 1
    #         else:
    #             fp[image_ids.index(d)] = 1.
    #     else:
    #         fp[image_ids.index(d)] = 1.
    print(tp)
    print(fp)
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    print('gt框个数')
    print(npos)
    rec = tp / float(npos)  # npos对
    # print(rec)
    print(tp)
    print(fp)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # print(prec)
    ap = voc_ap(rec, prec, use_07_metric)

    # plt.plot(rec, prec, lw=2,
    #          label='Precision-recall curve of class {} (area = {:.4f})'
    #                ''.format(cls, ap))
    # plt.plot(rec, prec, lw=2,
    #          label='Precision-recall curve of class {} (area = {:.4f})'
    #                ''.format('vehicle', ap))
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.grid(True)
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall')
    # plt.legend(loc="upper right")
    # plt.show()
    real_prec = tp[-1]/nd
    real_rec = tp[-1]/npos
    real_f1 = 2 * (real_prec * real_rec) / (real_prec + real_rec)
    print('precision:'+str(real_prec))
    print('recall:'+str(real_rec))
    print('f1:'+str(real_f1))
    return rec, prec, ap


if __name__ == '__main__':
    # rec, prec, ap = voc_eval(detpath='exp/PseudoDetection_coarse_init_DXB/results.json',  # todo
    #                          annopath=r'E:\Jiang\data\MOT\DXB\gt.json',  # todo
    #                          ovthresh=0.3)
    # print('----------------------')
    # print(rec)
    # print(prec)
    # print(ap)
    # results_DSFNet_model_best.json
    # E:\Jiang\data\MOT\DXB\bbox_centernet
    rec, prec, ap = voc_eval(detpath='exp/SkySat/results_24.json',  # todo
                             annopath='E:\Jiang\data\MOT\SkySat\\bbox_centernet\\gt.json',  # todo
                             ovthresh=0.3)
    # rec, prec, ap = voc_eval(detpath='exp/VISO/results_24.json',  # todo
    #                          annopath=r'E:\Jiang\data\RsCarData\gt.json',  # todo
    #                          ovthresh=0
    #                          )
    print('----------------------')
    # print(rec.mean())
    # print(prec.mean())
    print(ap)

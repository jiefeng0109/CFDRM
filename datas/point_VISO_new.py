import random

import numpy as np
import cv2
import time
import datetime
import os
from skimage import morphology
from skimage import measure
import shutil


def get_point_from_bbox(load_path, dataset):
    # 根据gt框随机中心点附近的一个点
    # 附近定义距离为-1-1
    # 生成点标记
    print(os.getcwd())
    with open('MOT\\VISO\\' + dataset +'output_real_point.txt', 'w') as f:
        output_real_point = []
        output_pseudo_point = []

        file_list = os.listdir(load_path)
        # print(len(file_list))
        data = np.loadtxt(r'E:\\Jiang\\data\\RsCarData\\images\\' + dataset + 'gt\gt.txt', delimiter=',', dtype=int).tolist()
        for i in range(1, len(file_list)+1):  # 图片索引从1开始的
            if i % 5 != 0:
                continue
            gt = [car for car in data if car[0] == i]
            for car in gt:
                # box_real = [car[2], car[3], car[4], car[5]]
                # 之后尝试randrange(car[2],car[4]),就是随机取点了
                # 这里其实可以做一个选择论证，有一篇论文论证过  # todo
                # print(box_real)
                center_real_x = car[2]+car[4]/2   # x+w/2
                center_real_y = car[3]+car[5]/2   # y+h/2
                # print(center_real_x, center_real_y)
                center_pseudo_x = random.uniform(center_real_x-1, center_real_x+1)
                center_pseudo_y = random.uniform(center_real_y-1, center_real_y+1)
                # print(center_pseudo_x, center_pseudo_y)
                output_real_point.append([car[0], car[1], center_real_x, center_real_y, car[6]])
                output_pseudo_point.append([car[0], car[1], center_pseudo_x, center_pseudo_y, car[6]])

        np.savetxt('MOT\\VISO\\' + dataset +'output_real_point.txt', np.array(output_real_point), delimiter=',', fmt='%d')
        np.savetxt('MOT\\VISO\\' + dataset +'output_pseudo_point.txt', np.array(output_pseudo_point), delimiter=',', fmt='%d')
    f.close()
    return output_pseudo_point


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


# datasets = ['train\\%03d\\' %(i) for i in range(1, 71)]
datasets = ['train\\%03d\\' %(i) for i in range(1, 71)]   # todo debug
for dataset in datasets:
    setDir(r'E:\Jiang\code\MOT\UnsupervisedDetection\datas\MOT\VISO\\' + dataset)
    load_path = 'E:\\Jiang\\data\\RsCarData\\images\\' + dataset + 'img1\\'

    get_point_from_bbox(load_path, dataset)
    # gt.txt包含更多图片的标注，但图片没有那么多，只有122张
    # print(len(output_pseudo_point))
    # load_path = r'D:\Liang\Jilin\SD\9590-2960\img'
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    model = cv2.createBackgroundSubtractorMOG2(history=3)  # todo 重要参数，一般越高召回率越大

    bboxes_video = []
    output = []
    data = np.loadtxt('MOT\\VISO\\' + dataset +'output_pseudo_point.txt', delimiter=',', dtype=int).tolist()
    # 加个0方便后面统计
    for i in range(len(data)):
        data[i].append(0)
        # print(data[i])
    # for car in data:
    #     print(car)
    for i, file in enumerate(os.listdir(load_path)):
        # 图片遍历
        print(i, file)
        # 隔5帧取一下   todo  重要参数
        if int(file[:-4]) % 5 != 0:
            continue
        frame = cv2.imread(os.path.join(load_path, file))
        gt = [car for car in data if car[0] == int(file[:-4])]  # todo
        print(len(gt))
        print(gt)
        # print(frame.shape[0])  # height,width,channel
        mask = model.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask = morphology.remove_small_objects(np.array(mask, dtype=bool), min_size=4, connectivity=2, in_place=True)
        mask = np.array(mask, dtype=int) * 255

        labeled = measure.label(mask)
        bboxes_frame = []
        pic_output = []
        # output_length = len(output)  # 记录遍历每一次图片后的长度，以方便对齐pic_output
        for j, region in enumerate(measure.regionprops(labeled)):
            # 伪标签框进行遍历
            y1, x1, y2, x2 = region.bbox
            # 用最原始的，因为这样伪标签生成较好一些,超参数
            box = [x1, y1, x2, y2]

            # frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            #                       (255, 0, 0), 1)
            # print(box)

            index_gt = []  # 保存在框里的点的索引
            num = 0
            # 遍历点标记
            # 图片一个框,遍历所有点
            for m in range(len(gt)):
                # print(i)
                # print(gt[i])
                # print(car[2], car[3], car[5])
                # # 如果在里面则用传统算法的伪标签框
                # print(box)
                if box[0] < gt[m][2] < box[2] and box[1] < gt[m][3] < box[3]:
                    print('in')
                    print(gt[m][2], gt[m][3])
                    print(box)
                    # output.append([
                    #     i, j, max(x1 - 3, 0), max(y1 - 3, 0), min(x2 + 3, frame.shape[1]), min(y2 + 3, frame.shape[0])])
                    index_gt.append(m)  # 统计
            print(index_gt)

            # 一个框对应一个点
            if len(index_gt) == 1:
                print("1 box vs 1 point")
                if gt[index_gt[0]][5] == 1:
                    pass
                else:
                    gt[index_gt[0]][5] = 1
                    # output.append([gt[index_gt[0]][0],
                    #                gt[index_gt[0]][1],
                    #                max(x1 - 3, 0),
                    #                max(y1 - 3, 0),
                    #                min(x2 + 3, frame.shape[1]),
                    #                min(y2 + 3, frame.shape[0]),
                    #                gt[index_gt[0]][2],
                    #                gt[index_gt[0]][3],
                    #                ])
                    pic_output.append([gt[index_gt[0]][0],
                                       gt[index_gt[0]][1],
                                       max(x1, 0),
                                       max(y1, 0),
                                       min(x2, frame.shape[1]),
                                       min(y2, frame.shape[0]),
                                       gt[index_gt[0]][2],
                                       gt[index_gt[0]][3],
                                       ])

            elif len(index_gt) > 1:
                print("1 box vs multi points")
                # 一个框对应多个点,那么按照框内取最近邻边作为伪标签框
                # 其实这里存在两个点都对应一个框的一个象限里面，就会有问题，就会出现一个框把另一个框包裹住
                if gt[index_gt[0]][5] == 1:
                    pass
                else:
                    for index in index_gt:
                        gt[index][5] = 1
                        w_pseudo = min(gt[index][2]-box[0], box[2]-gt[index][2])*2  # 最近的一边作为一半的w,h
                        h_pseudo = min(gt[index][3]-box[1], box[3]-gt[index][3])*2  # 最近的一边作为一半的w,h

                        pic_output.append(
                            [gt[index][0],
                             gt[index][1],
                             max(gt[index][2]-w_pseudo/2, 0),
                             max(gt[index][3]-h_pseudo/2, 0),
                             min(gt[index][2]+w_pseudo/2, frame.shape[1]),
                             min(gt[index][3]+h_pseudo/2, frame.shape[0]),
                             gt[index][2],
                             gt[index][3],
                             ])

            print('-------------------------------------------------------')
            print(len(output))
        print(gt)
        print(len(gt))

        # 可能存在一个点对应多个框的情况，只取第一个，后续可以都保存，然后取一个与伪标签框中心距离最小的框
        # attention 此处不用管car[5],因为car[5]终究是保留了一个框，所以还是1没问题.
        # for car in gt:
        #     if car[5] == 1:
        #
        #         pass
        print(len(pic_output))
        del_index_pic_out = []
        for num1 in range(len(pic_output)):
            for num2 in range(num1+1, len(pic_output)):
                if num1 in del_index_pic_out or num2 in del_index_pic_out:
                    continue
                else:
                    if pic_output[num1][6] == pic_output[num2][6] and pic_output[num1][7] == pic_output[num2][7]:
                        pseudo_bbox1_center_x = (pic_output[num1][2]+pic_output[num1][4])/2
                        pseudo_bbox1_center_y = (pic_output[num1][3]+pic_output[num1][5])/2
                        pseudo_bbox2_center_x = (pic_output[num2][2]+pic_output[num2][4])/2
                        pseudo_bbox2_center_y = (pic_output[num2][3]+pic_output[num2][5])/2
                        dis_1 = (pic_output[num1][6]-pseudo_bbox1_center_x)**2 + (pic_output[num1][7]-pseudo_bbox1_center_y)**2
                        dis_2 = (pic_output[num2][6]-pseudo_bbox2_center_x)**2 + (pic_output[num2][7]-pseudo_bbox2_center_y)**2
                        if dis_1 <= dis_2:
                            # del pic_output[num2]
                            del_index_pic_out.append(num2)
                        else:
                            del_index_pic_out.append(num1)
                        # del pic_output[num1]
        print(del_index_pic_out)
        # pic_output=[n for i, n in enumerate(pic_output) if i not in del_index_pic_out]
        del_index_pic_out.sort()
        del_index_pic_out.reverse()
        # print(sorted(del_index_pic_out).reverse())  # 对索引进行反转，使其从后往前删除
        # print(sorted(del_index_pic_out))

        for del_index in del_index_pic_out:
            pic_output.pop(del_index)
        # 统计图片内内所有生成的伪标签框的大小的平均值和方差，然后3sigma原则随机取
        x = 0
        y = 0
        for car in gt:
            if car[5] == 0:
                x += 1

                # 默认点标记是中心点
                # 统计图片内内所有生成的伪标签框的大小的平均值和方差，然后3sigma原则随机取
                bbox_sigma = 0
                bbox_mean = 0
                # 需要统计宽和长，因为不一定是正框
                bbox_length = []
                bbox_width = []
                print(len(pic_output))
                for bbox in pic_output:
                    bbox_length.append(bbox[4] - bbox[2])
                    bbox_width.append(bbox[5] - bbox[3])
                if len(bbox_width) == 0:
                    # 说明没有预测到对应的伪标签框，直接暴力初始化
                    # 暴力直接加长宽都是4的正框
                    output.append([
                        car[0],
                        car[1],
                        max(car[2] - 2, 0),
                        max(car[3] - 2, 0),
                        min(car[2] + 2, frame.shape[1]),
                        min(car[3] + 2, frame.shape[0]),
                        car[2],
                        car[3],
                    ])
                elif len(bbox_width) != 0:
                    # 设定法则的左右边界
                    # print(bbox_width)
                    # 设定法则的左右边界
                    left = np.mean(bbox_length) - 3 * np.std(bbox_length)
                    right = np.mean(bbox_length) + 3 * np.std(bbox_length)
                    # print(left, right)  # 一般是7-15，但是无监督生成的一般是9-14
                    pseudo_bbox_length = np.random.uniform(max(left, 4), min(right, 8))
                    # print(int(pseudo_bbox_length))
                    # print(pseudo_bbox_length)
                    # print(max(car[2] - pseudo_bbox_length / 2, 0))
                    pseudo_bbox_width = np.random.uniform(max(np.mean(bbox_width) - 3 * np.std(bbox_width), 4),
                                                          min(np.mean(bbox_width) + 3 * np.std(bbox_width), 8))
                    pic_output.append([
                        car[0],
                        car[1],
                        max(car[2] - pseudo_bbox_length / 2, 0),
                        max(car[3] - pseudo_bbox_width / 2, 0),
                        min(car[2] + pseudo_bbox_length / 2, frame.shape[1]),
                        min(car[3] + pseudo_bbox_width / 2, frame.shape[0]),
                        car[2],  # pseudo bbox
                        car[3]
                    ])  # int默认向下取整，四舍五入更好一些,可以＋0.5
                # # 暴力直接加长宽都是10的正框
                # output.append([
                #     i, j, max(car[2] - 5, 0), max(car[3] - 5, 0),
                #     min(car[2] + 5, frame.shape[1]), min(car[3] + 5, frame.shape[0])])
            elif car[5] != 0:
                y += 1
        # print(x)
        # print(y)
        # print(len(pic_output))
        # 最后才加
        for pic_out in pic_output:
            output.append(pic_out)

        print(len(output))

        print('#########################################################################')
            # car[5] = num_in_bbox
            # num_in_bbox = 0
            # for car in gt:
            #     print(car)
            #     if car[5] == 0:
            #         print('out')
            #         output.append(
            #             [i, j, max(car[2] - 5, 0), max(car[3] - 5, 0), min(car[2] + 5, frame.shape[1]),
            #              min(car[3] + 5, frame.shape[0])])

        # cv2.imwrite('GMM\\' + file, frame)
        bboxes_video.append(bboxes_frame)
        # print('GMM\\' + file)

    with open('MOT\\VISO\\' + dataset + 'pseudo_bbox.txt', 'w') as f:
        # 这里的j有问题的，后面没用
        np.savetxt('MOT\\VISO\\' + dataset + 'pseudo_bbox.txt', np.array(output), delimiter=',', fmt='%d')  # todo
    f.close()



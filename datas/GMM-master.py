import numpy as np
import cv2
import time
import datetime
import os
from skimage import morphology
from skimage import measure
datasets = ['DXB\\1460-1400\\', 'SD\\9590-2960\\', 'SkySat\\002\\']
dataset = datasets[2]
load_path = 'E:\\jqp\\data\\MOT\\' + dataset + 'img\\'
# load_path = r'D:\Liang\Jilin\SD\9590-2960\img'
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
model = cv2.createBackgroundSubtractorMOG2(history=9)

bboxes_video = []
output = []

for i, file in enumerate(os.listdir(load_path)):
    print(i, file)

    frame = cv2.imread(os.path.join(load_path, file))
    # print(frame.shape[0])  # height,width,channel
    mask = model.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = morphology.remove_small_objects(np.array(mask, dtype=bool), min_size=4, connectivity=2, in_place=True)
    mask = np.array(mask, dtype=int) * 255

    labeled = measure.label(mask)
    bboxes_frame = []
    for j, region in enumerate(measure.regionprops(labeled)):
        y1, x1, y2, x2 = region.bbox
        box = [x1 - 3, y1 - 3, x2 + 3, y2 + 3]
        frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                              (255, 0, 0), 1)
        bboxes_frame.append(box)
        output.append([i, j, max(x1 - 3, 0), max(y1 - 3, 0), min(x2 + 3, frame.shape[1]), min(y2 + 3, frame.shape[0])])

    cv2.imwrite('GMM\\' + file, frame)
    bboxes_video.append(bboxes_frame)
    print('GMM\\' + file)

print(output[0])
# 这里的j有问题的，后面没用
np.savetxt('gt.txt', np.array(output[1:]), delimiter=',', fmt='%d')

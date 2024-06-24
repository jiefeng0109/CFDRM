# w = 11.439513061120973 h = 11.829951813340097

import numpy as np
import json
import os
import matplotlib.pyplot as plt

with open(r'E:\jqp\data\MOT\SkySat\pseudo.json', 'r') as rf:
    data = json.load(rf)

rank = []
for file in os.listdir(r'E:\jqp\data\MOT\SkySat\train'):
    # print(int(file[0:3]))
    objs = [i for i in data['annotations'] if i['image_id'] == int(file[0:3])]
    print(objs)
    ave_w, ave_h, ave_a = [], [], []
    for obj in objs:
        ave_w.append(obj['bbox'][2])
        ave_h.append(obj['bbox'][3])
        ave_a.append(obj['bbox'][2] * obj['bbox'][3])
    rank.append(np.std(np.array(ave_a), ddof=1))  # 无偏样本标准差ddof=1
    print(rank)
    # 面积求个方差
    # 方差越小越好吧，框越相近

print(min(rank), max(rank))
Q = []
MIN_Q = min(rank)
MAX_Q = max(rank)
for i in rank:
    Q.append(1 - ((i - MIN_Q) / (MAX_Q - MIN_Q)))  # 1-归一化
print(Q)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
import json
import random
import cv2


def crop_img(load_path=r'D:\Liang\Jilin\SD',
             save_path=r'D:\Liang\Jilin\SD',
             # datasets=['1960-2690','250-2970'],
             # datasets=['1460-1400'],
             # datasets=['4570-3360', '8450-2180'],
             datasets=['9590-2960'],
             cut_size=500,
             stride=500,
             ):
    save_id = 1
    save_label = []

    for dataset in datasets:
        file_list = os.listdir(os.path.join(load_path, dataset, 'img'))
        data = np.loadtxt(os.path.join(load_path, dataset, 'gt.txt'), delimiter=',', dtype=int).tolist()
        for file in file_list[1:-1]:
            img_id = int(file[:-4])
            label = [i for i in data if i[0] == img_id]
            img_pre = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id - 1)))
            img = Image.open(os.path.join(load_path, dataset, 'img', file))
            img_post = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id + 1)))
            img_w, img_h = img.size
            for obj in label:
                w, h = obj[4] - obj[2], obj[5] - obj[3]
                # crop_x1, crop_y1 = max(int(obj[2] - w / 4), 0), max(int(obj[3] - h / 4), 0)
                # crop_x2, crop_y2 = min(int(obj[4] + w / 4), img_w), min(int(obj[5] + h / 4), img_h)
                crop_x1, crop_y1 = max(int(obj[2]), 0), max(int(obj[3]), 0)
                crop_x2, crop_y2 = min(int(obj[4]), img_w), min(int(obj[5]), img_h)
                crop_area = [crop_x1, crop_y1, crop_x2, crop_y2]
                img_crop = img.crop(crop_area)
                img_pre_crop = img_pre.crop(crop_area)
                img_post_crop = img_post.crop(crop_area)
                img_save = np.zeros((32, 32, 9))
                img_save[:, :, 0:3] = cv2.resize(np.array(img_pre_crop), (32, 32))
                img_save[:, :, 3:6] = cv2.resize(np.array(img_crop), (32, 32))
                img_save[:, :, 6:9] = cv2.resize(np.array(img_post_crop), (32, 32))
                np.save(os.path.join(save_path, 'vehicle\%05d_%d.npy' % (save_id, 1)),
                        np.array(img_save, dtype=int))
                save_id += 1
                img_save = np.zeros((32, 32, 9))
                img_save[:, :, 0:3] = cv2.resize(np.array(img_crop), (32, 32))
                img_save[:, :, 3:6] = cv2.resize(np.array(img_crop), (32, 32))
                img_save[:, :, 6:9] = cv2.resize(np.array(img_crop), (32, 32))
                np.save(os.path.join(save_path, 'vehicle\%05d_%d.npy' % (save_id, 0)),
                        np.array(img_save, dtype=int))
                save_id += 1
                if random.random() < 0.2:
                    rand_x, rand_y = random.randint(0, img_w - 18), random.randint(0, img_h - 18)
                    img_crop = img.crop([rand_x, rand_y, rand_x + 18, rand_y + 18])
                    img_pre_crop = img_pre.crop([rand_x, rand_y, rand_x + 18, rand_y + 18])
                    img_post_crop = img_post.crop([rand_x, rand_y, rand_x + 18, rand_y + 18])
                    img_save = np.zeros((32, 32, 9))
                    img_save[:, :, 0:3] = cv2.resize(np.array(img_pre_crop), (32, 32))
                    img_save[:, :, 3:6] = cv2.resize(np.array(img_crop), (32, 32))
                    img_save[:, :, 6:9] = cv2.resize(np.array(img_post_crop), (32, 32))
                    np.save(os.path.join(save_path, 'vehicle\%05d_%d.npy' % (save_id, 0)),
                            np.array(img_save, dtype=int))
                    save_id += 1

            print(os.path.join(load_path, dataset, 'img', file))
    print(save_id)
    return save_id


if __name__ == "__main__":
    save_label = crop_img()

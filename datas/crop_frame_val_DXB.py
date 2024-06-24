import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
import json


def crop_img(load_path=r'E:\Jiang\data\MOT\\',
             save_path=r'E:\Jiang\data\MOT\DXB',  # todo save_path
             # datasets=['1960-2690','250-2970'],
             # datasets=['1460-1400'],
             # datasets=['4570-3360', '8450-2180'],
             datasets=['DXB\\1460-1400\\'],  # todo datasets
             cut_size=500,  # todo
             stride=500,
             ):
    save_id = 1
    save_label = []

    for dataset in datasets:
        file_list = os.listdir(os.path.join(load_path, dataset, 'img'))
        # data = np.loadtxt(os.path.join(load_path, dataset, 'gt.txt'), delimiter=',', dtype=int).tolist()
        # GMM生成的
        # data = np.loadtxt(r'pseudo_bbox.txt', delimiter=',', dtype=int).tolist()  # todo np.loadtxt
        data = np.loadtxt(r'E:\Jiang\data\MOT\DXB\1460-1400\gt\gt.txt', delimiter=',', dtype=int).tolist()
        # print(data)
        for file in file_list[1:-1]:
            img_id = int(file[:-4])
            label = [i for i in data if i[0] == img_id]
            # print(label)

            img_pre = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id - 1)))
            img = Image.open(os.path.join(load_path, dataset, 'img', file))
            img_post = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id + 1)))
            img_w, img_h = img.size

            for i in range(img_w // stride):  # 这里可能会有错
                for j in range(img_h // stride):
                    cut_x = stride * i
                    cut_y = stride * j
                    print(cut_x, cut_y)
                    for obj in label:
                        # 位于切割处的bbox会被舍弃，存在问题
                        print(obj)
                        if obj[2] >= cut_x and obj[3] >= cut_y \
                                and obj[4] <= cut_x + cut_size and obj[5] <= cut_y + cut_size:
                            # 相对crop的图片的位置下x,y
                            # w,h绝对长度
                            print('####################')
                            x, y = obj[2] - cut_x, obj[3] - cut_y
                            w, h = obj[4] - obj[2], obj[5] - obj[3]
                            save_label.append([x, y, w, h, save_id])
                            # print(save_label)
                    img_crop = img.crop([cut_x, cut_y, cut_x + cut_size, cut_y + cut_size])
                    img_pre_crop = img_pre.crop([cut_x, cut_y, cut_x + cut_size, cut_y + cut_size])
                    img_post_crop = img_post.crop([cut_x, cut_y, cut_x + cut_size, cut_y + cut_size])

                    img_save = np.zeros((cut_size, cut_size, 9))
                    img_save[:, :, 0:3] = img_pre_crop
                    img_save[:, :, 3:6] = img_crop
                    img_save[:, :, 6:9] = img_post_crop

                    np.save(os.path.join(save_path,
                                         'val\%03d_%s_%d_%d_%d.npy' % (save_id, dataset[4:-1], img_id, cut_x, cut_y)),  # todo dataset
                            np.array(img_save, dtype=int))
                    save_id += 1
                    # save_id表示的切割之后的图片的id
                    print(save_id)
            # print(os.path.join(load_path, dataset, 'img', file))
    print(save_id, len(save_label))
    return save_label, save_id


def convert_json(data, save_path, img_size=500):  # todo
    coco = dict()
    coco['info'] = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []
    coco["licenses"] = []

    info = {"year": 2022.4,
            "version": '1.0',
            "description": 'UnsupervisedDetection',
            "contributor": 'liang',
            "url": 'null',
            "date_created": '2022.4.23'
            }
    coco['info'] = info

    categories = {"id": 1,
                  "name": 'car',
                  "supercategory": 'null',
                  }
    coco['categories'].append(categories)

    # Q = [0.2109456350130382, 0.4164768129301495, 0.7806071426098685, 0.0, 0.4890749916827972, 0.8101187594505448,
    #      0.9134212768148966, 0.533238997556897, 0.5228082815937627, 0.7005179593152957, 0.8710986721634774,
    #      0.6309504947911984, 0.5152374500460731, 0.7240636873643034, 0.9108665663073833, 0.6612578262733948,
    #      0.5055797513091478, 0.6569365370563904, 0.9135318145606179, 0.6752043211595049, 0.5417686212279262,
    #      0.6267050261547835, 0.8393698021233162, 0.636291272058844, 0.5734570696817336, 0.6765822392288287,
    #      0.7046928220712776, 0.6834057953426866, 0.5175564558818768, 0.6821085632591197, 0.9078229855245391,
    #      0.6449698836107542, 0.624655198006133, 0.6379636298811299, 0.6621687668839272, 0.6087753790384363,
    #      0.5877228437479833, 0.6642831657993264, 0.6576661535060706, 0.6944563237095329, 0.5245345254415721,
    #      0.6031488987296189, 0.8198165826688781, 0.4097129846947023, 0.552643844119223, 0.6878296459717592,
    #      0.7967345057999641, 0.5379961364989496, 0.5539634503753198, 0.6802658054508159, 0.8004657842980464,
    #      0.6249518587341343, 0.6583838376996594, 0.7114624827895757, 0.8282065566921024, 0.6335722858695378,
    #      0.6725255143916689, 0.7187215901244519, 0.8250479984482035, 0.47033383222953695, 0.5614582658410356,
    #      0.6833055280486235, 0.9293539227621278, 0.5792722699079148, 0.6133846567812393, 0.6443712059053606,
    #      0.8479185833583586, 0.40537703427098326, 0.6544845837528119, 0.7358148948886807, 0.8384265158549722,
    #      0.6249340007392146, 0.6703836164076569, 0.6307327766726355, 0.7682848427607237, 0.5288255799929669,
    #      0.7033371272051849, 0.648059549874799, 0.5040826536643399, 0.5921155724741509, 0.7072587529717149,
    #      0.7538643691580598, 0.7930124443311655, 0.36985777013380183, 0.6846368872539212, 0.5425460030419594,
    #      0.8655210963103145, 0.48294024263162816, 0.7125831398598108, 0.7234357334531829, 0.873214410902275,
    #      0.41538694037746016, 0.7059109132849914, 0.6884402063214372, 0.8073348690383768, 0.41290069120770456,
    #      0.6665647764112406, 0.7465172970787544, 1.0, 0.4311368546408696, 0.694243497125129, 0.7966383084742196,
    #      0.899483675731048, 0.4739429066177878, 0.5838106379579163, 0.6327042716985178, 0.8630133873125683,
    #      0.47293956866802356, 0.5184073456869107, 0.7044232060989952, 0.9675000996747076, 0.3771631753851956,
    #      0.5870022087219903, 0.7122746248447641, 0.7868694435603023, 0.2914391139938136, 0.6739598202715249,
    #      0.683960104371609, 0.835184644155835, 0.5883546906894374, 0.6711616170193804, 0.7566517454951864,
    #      0.8916634929028733, 0.3936903856428319, 0.6467623485838943, 0.7318770314104822, 0.9250166528019146,
    #      0.6045750301708762, 0.569970334474283, 0.7190761178984384, 0.8377333149817989, 0.5280899163208184,
    #      0.775704863178482, 0.7356802476815179, 0.8299435496441674, 0.5460477800281527, 0.6952020341904306,
    #      0.6910506449232751, 0.7993837871231968, 0.5015949869918748, 0.7381845726811469, 0.747775982603833,
    #      0.8593212713099556, 0.37341691393299437, 0.6898912100541607, 0.8148897914512612, 0.8589005317023609,
    #      0.6389997426042595, 0.6514367663570095, 0.5327170692577314, 0.8172543664408353, 0.5862132169851088,
    #      0.7625865890017938, 0.5987283865726725, 0.7827561209881155, 0.15811821061202147, 0.7021967199297326,
    #      0.541685210565646, 0.8259678108714064, 0.43093385087783465, 0.789753112332349, 0.7705903797371814,
    #      0.796569243789116, 0.5341662876915092, 0.6852173391979722, 0.7808211416291934, 0.8662441903757604,
    #      0.5185877606375671, 0.6280690258045284, 0.7991595398237572, 0.9215703542350863, 0.43407872740853015,
    #      0.5214328533048904, 0.8011292728629017, 0.8568713711979138, 0.6399960424525732, 0.7484595361776425,
    #      0.8373585190593069, 0.9132075296897079, 0.5763746879616773, 0.6787795528441842, 0.8118368669735789,
    #      0.8209040795354157, 0.456482109095251, 0.7755948826144207, 0.7009552175097232, 0.9975181006097859,
    #      0.439211892156694, 0.7802563114196734, 0.831749220416178, 0.7694867677771564, 0.5605174235128197,
    #      0.7680799475098173, 0.7703341829325725, 0.7288497322263894, 0.5790779485183946, 0.72592770776049,
    #      0.8120783598846494, 0.7207366469298977, 0.3940241385490474]

    for file in os.listdir(os.path.join(save_path, 'val')):  # todo
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

    for i, item in enumerate(data):
        print(len(data))
        print(i, item)
        # bbox[] is x,y,w,h
        print(item[4])
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

    with open(os.path.join(save_path, 'gt.json'), 'w') as f:  # todo
        json.dump(coco, f)

    return coco


if __name__ == "__main__":

    save_label, img_num = crop_img()
    coco_json = convert_json(data=save_label, save_path=r'E:\Jiang\data\MOT\DXB')  # todo

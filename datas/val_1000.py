import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
import json


def crop_img(load_path=r'D:\Liang\Jilin\Skysat',
             save_path=r'D:\Liang\Jilin\Skysat',
             datasets=['001'],
             ):
    save_id = 1
    save_label = []

    for dataset in datasets:
        file_list = os.listdir(os.path.join(load_path, dataset, 'img'))
        # data = np.loadtxt(os.path.join(load_path, dataset, 'gt.txt'), delimiter=',', dtype=int).tolist()
        data = np.loadtxt('gt.txt', delimiter=',', dtype=int).tolist()
        for file in file_list[1:-1]:
            img_id = int(file[:-4])
            label = [i for i in data if i[0] == img_id]
            img_pre = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id - 1)))
            img = Image.open(os.path.join(load_path, dataset, 'img', file))
            img_post = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id + 1)))
            img_w, img_h = img.size
            for obj in label:
                x, y = obj[2], obj[3]
                w, h = obj[4] - obj[2], obj[5] - obj[3]
                save_label.append([x, y, w, h, save_id])
            img_save = np.zeros((img_h, img_w, 9))
            img_save[:, :, 0:3] = img_pre
            img_save[:, :, 3:6] = img
            img_save[:, :, 6:9] = img_post
            np.save(os.path.join(save_path, 'val\%03d_%s_%d.npy' % (save_id, dataset, img_id)),
                    np.array(img_save, dtype=int))
            save_id += 1
            print(os.path.join(load_path, dataset, 'img', file))
    print(save_id, len(save_label))
    return save_label, save_id


def convert_json(data, save_path, img_size=1000):
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
            "date_created": '2022.4.11'
            }
    coco['info'] = info

    categories = {"id": 1,
                  "name": 'car',
                  "supercategory": 'null',
                  }
    coco['categories'].append(categories)

    Q = [0.6107112170191685, 0.8778680534626218, 0.5170141207434489, 0.042839117212475464, 0.8846640898257618,
         0.7292015401837346, 0.7412257173208454, 0.4588026802523655, 0.7909737776924763, 0.9362946517230133,
         0.9132153236753463, 0.176189373586369, 0.621106090721735, 0.8084513698328146, 0.9140658498329954,
         0.856195457607127, 0.5790681961749737, 0.41689372994303786, 0.5037513102473952, 0.3199431873317494,
         0.019254434328450976, 0.7982519043750169, 0.9570723502760948, 0.40449546708374806, 0.9570272071341779,
         0.9675608069251753, 0.9117149292441069, 0.4815360868452998, 0.8080399175038397, 0.7744332293906072, 0.0,
         0.8694506636141989, 0.7661035063096822, 0.7172691571931562, 0.7758068519800203, 0.2664386723662271,
         0.8425406301201619, 0.8535577942235909, 0.5791117484978532, 0.04459379389544216, 0.793125434767332,
         0.7549575414590568, 0.8372042234890416, 0.17316503241331316, 0.7937694353582245, 0.817285355035907,
         0.8874372666903925, 0.31777418732096774, 0.8919744948060878, 0.39548091100692084, 0.8812301973673368,
         0.7235907196047917, 0.7719143593952167, 0.09805859546830775, 0.843155275214709, 0.6996232478323261,
         0.7791038394107838, 0.743543369905592, 0.780652792764832, 0.8594807411435219, 0.9111397499631798,
         0.7621656377521351, 0.8863921263889336, 0.9380199759212788, 0.9736029908876259, 0.6072938294454735,
         0.9731494423252791, 0.6030328700537644, 0.9358783053004629, 0.8969021951224554, 0.8935270387783198,
         0.6026683558137527, 0.9278359394813017, 0.8856983498651534, 0.631898391427536, 0.9296721315849727,
         0.9063980355572478, 0.6710475645780789, 0.9301910220660599, 1.0, 0.2831919660930381, 0.971422256460221,
         0.9144732167331939, 0.16310564520771276, 0.8317010795484034, 0.5550638457093884, 0.3585326309426168,
         0.804266253214989, 0.924460223003865, 0.856104402233583, 0.2160874481465822, 0.8269875438840323,
         0.13770167251398258, 0.7931733266455945, 0.2598085523043102, 0.8467463001513357, 0.7816318665299185,
         0.7044219508070402, 0.4093791409638352, 0.7573804463287793, 0.870916529904014, 0.8574377466033217,
         0.899816753042991, 0.7903320841730849, 0.651909486943667, 0.47736556558798093, 0.821820947962048,
         0.833455490006856, 0.7793396090346552, 0.8271413165926962, 0.7566043078376592, 0.48147290975682766,
         0.680585600868026, 0.7012000512258156, 0.8421368162018101, 0.692365660513125, 0.931300199939059,
         0.5892465171492781, 0.8763849447717089, 0.8221131383532779]

    for file in os.listdir(os.path.join(save_path, 'val')):
        if '001' in file:
            w, h = 400, 400
        else:
            w, h = 400, 600
        image = {"id": int(file.split('_')[0]),
                 "width": w,
                 "height": h,
                 "file_name": file,
                 "license": 0,
                 "flickr_url": 'null',
                 "coco_url": 'null',
                 "date_captured": '2022.4.11'
                 }
        coco['images'].append(image)

    for i, item in enumerate(data):
        # bbox[] is x,y,w,h
        bbox = [item[0], item[1], item[2], item[3]]
        annotation = {"id": i,
                      "image_id": item[4],
                      "category_id": 1,
                      "segmentation": [],
                      "area": item[2] * item[3],
                      "bbox": bbox,
                      "iscrowd": 0,
                      'ignore': 0,
                      'weight': Q[item[4] - 1],
                      }
        coco['annotations'].append(annotation)
    print(len(coco['images']), len(coco['annotations']))

    with open(os.path.join(save_path, 'pseudo.json'), 'w') as f:
        json.dump(coco, f)

    return coco


if __name__ == "__main__":
    save_label, img_num = crop_img()
    coco_json = convert_json(data=save_label, save_path=r'D:\Liang\Jilin\Skysat')

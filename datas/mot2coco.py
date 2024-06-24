import os
import json

img_size = 1000
save_path = r'D:\Liang\Jilin\DXB'
annotations = []
f = open(r'D:\Liang\Jilin\DXB\1460-1400\gt.txt', 'r')
for line in f:
    line = line.split(',')
    line = [int(i) for i in line]
    if 0 < line[0] <= 50:
        annotations.append([line[0], line[2], line[3], line[4], line[5]])
f.close()

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

for file in os.listdir(r'D:\Liang\Jilin\DXB\1460-1400\img')[1:-1]:
    image = {"id": int(file[:-4]),
             "width": img_size,
             "height": img_size,
             "file_name": file,
             "license": 0,
             "flickr_url": 'null',
             "coco_url": 'null',
             "date_captured": '2022.4.11'
             }
    coco['images'].append(image)

for i, obj in enumerate(annotations):
    # bbox[] is x,y,w,h
    bbox = [obj[1], obj[2], obj[3] - obj[1], obj[4] - obj[2]]
    annotation = {"id": i,
                  "image_id": obj[0],
                  "category_id": 1,
                  "segmentation": [],
                  "area": bbox[2] * bbox[3],
                  "bbox": bbox,
                  "iscrowd": 0,
                  'ignore': 0
                  }
    coco['annotations'].append(annotation)

print(len(coco['images']), len(coco['annotations']))

with open(os.path.join(save_path, 'annotation.json'), 'w') as f:
    json.dump(coco, f)

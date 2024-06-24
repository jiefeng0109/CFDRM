import os
import json
import xml.etree.ElementTree as ET


def get(root, name):
    return root.findall(name)


def parser(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
root_path = r'D:\Liang\VISO\validation data'
xml_lists = [i for i in os.listdir(root_path) if '.' not in i]
categories = {'car': 1, 'airplane': 2, 'ship': 3, 'train': 4}
img_id = 0
obj_id = 0

for xml_list in xml_lists:
    for line in os.listdir(os.path.join(root_path, xml_list, 'xml')):
        tree = ET.parse(os.path.join(root_path, xml_list, 'xml', line))
        root = tree.getroot()
        filename = os.path.basename(line)[:-4] + ".jpg"
        size = parser(root, 'size', 1)
        width = int(parser(size, 'width', 1).text)
        height = int(parser(size, 'height', 1).text)
        image = {'file_name': '%s\img1\%s' % (xml_list, filename), 'height': height, 'width': width, 'id': img_id}
        json_dict['images'].append(image)
        img_id += 1
        for obj in get(root, 'object'):
            category = parser(obj, 'name', 1).text
            category_id = categories[category]
            bndbox = parser(obj, 'bndbox', 1)
            xmin = int(float(parser(bndbox, 'xmin', 1).text))
            ymin = int(float(parser(bndbox, 'ymin', 1).text))
            xmax = int(float(parser(bndbox, 'xmax', 1).text))
            ymax = int(float(parser(bndbox, 'ymax', 1).text))
            obj_width = abs(xmax - xmin)
            obj_height = abs(ymax - ymin)
            annotation = {'area': obj_width * obj_height,
                          'iscrowd': 0,
                          'image_id': img_id,
                          'bbox': [xmin, ymin, obj_width, obj_height],
                          'category_id': category_id,
                          'id': obj_id,
                          'ignore': 0,
                          'segmentation': []}
            json_dict['annotations'].append(annotation)
            obj_id = obj_id + 1
    print(xml_list)

for cate, cid in categories.items():
    cat = {'supercategory': 'none', 'id': cid, 'name': cate}
    json_dict['categories'].append(cat)
json_fp = open(os.path.join(root_path, 'val.json'), 'w')
json_str = json.dumps(json_dict)
json_fp.write(json_str)
json_fp.close()

import json


def rank_bbox(annopath):
    """
    rank by weight
    """

    with open(annopath, 'r') as rf:
        annotations = json.load(rf)  # gt

    imagenames = [i['file_name'] for i in annotations['images']]
    # print(imagenames)
    # 对每张图片的xml获取函数指定类的bbox等

    # print('-----------------------')
    all_weights = []
    for img_id, imagename in enumerate(imagenames):
        # print('-----------')
        # print(img_id+1)
        # print(imagename)
        R = [obj for obj in annotations['annotations'] if obj['image_id'] == img_id + 1]
        # print(R)
        # print(len(R))
        weights = [x['weight'] for x in R]
        all_weights.append(weights)
    new_weights = []
    for weight in all_weights:
        # print(weight)
        new_weights.extend(weight)
    # print(type(new_weights))
    # print(new_weights)
    new_weights.sort()
    # print(new_weights)
    # print(len(new_weights))
    # print(new_weights[int(len(new_weights)/3)])
    # print(new_weights[int(len(new_weights)/3*2)])
    weight_1 = new_weights[int(len(new_weights)/3)]  # 0.505
    weight_2 = new_weights[int(len(new_weights)/3*2)]  # 0.633
    return weight_1, weight_2


if __name__ == '__main__':

    # rank_bbox(annopath=r'E:\Jiang\data\MOT\SD\pseudo_label\pseudo_0.json')
    rank_bbox(annopath=r'E:\Jiang\data\RsCarData\pseudo_0.json')


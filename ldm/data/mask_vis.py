"""
34.jpg
source_img = "/opt/ml/code/leixiaoning/datasets/data_YC4/MFD_Detect_PSL_YC4_Val_20231023_YC4_coco/data/MixImages/NG_043146_001CB31000000DD7R0312060.png"
target_img = "/opt/ml/code/leixiaoning/datasets/data_YC4/MFD_Detect_PSL_YC4_Val_20230727_YC4_coco/data/MixImages/NG_055054_001CB23000000BD7N1909605.png"
33.jpg
source_img = "/opt/ml/code/leixiaoning/datasets/data_YC4/MFD_Detect_PSL_YC4_Val_20231023_YC4_coco/data/MixImages/NG_041814_001CB31000000DD7N0431404.png"
target_img = "/opt/ml/code/leixiaoning/datasets/data_YC4/MFD_Detect_PSL_YC4_Val_20230727_YC4_coco/data/MixImages/NG_053602_001CB23000000BD7R2016134.png"

32.jpg
source_img = "datasets/yc4/val20231023/MixImages/NG_002001_001CB31000000DD7R0416162.png"
target_img = "datasets/yc4/val20230727/MixImages/NG_001503_001CB23000000BD7S2005882.png"

31.jpg
source_img = "/opt/ml/code/leixiaoning/datasets/data_YC4/MFD_Detect_PSL_YC4_Val_20231023_YC4_coco/data/MixImages/NG_023553_001CB31000000DD7R0422500.png"
target_img = "/opt/ml/code/leixiaoning/datasets/data_YC4/MFD_Detect_PSL_YC4_20230516_YC4_coco/data/MixImages/102922_1683705166-0_.png"
"""


# from detectron2.data.datasets.coco import load_coco_json
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog
# import numpy as np
# import cv2

#ppm
# name = 'wgj_train_green'
# dataset_dicts = load_coco_json('datasets/waiguanji/green_train/annotations.utf.json',
#                                 'datasets/waiguanji/green_train',
#                                 name)

# from pycocotools.coco import COCO
# coco_api = COCO('24_04_18_08_46_07_781-1-001CEBV0000002E4B2702590-2001_left.json')


import os
import json
import cv2
import numpy as np
import pycocotools.mask as mask_util
from visualizer import GenericMask


def _convert_masks(masks_or_polygons, height, width):
    m = masks_or_polygons
    ret = []
    for x in m:
        if isinstance(x, GenericMask):
            ret.append(x)
        else:
            ret.append(GenericMask(x, height, width))
    return ret

# load json
dataset_path = '/opt/ml/code/xuxichen/datasets/blue-dataset/'

_file_names = os.listdir(dataset_path)
file_names = []
for file_name in _file_names:
    file_name = file_name.split('.')
    if file_name[1] == "json":
        file_names.append(file_name[0])
print("anomaly image number: {}".format(len(file_names)))

# load label
labels = []
print('labels:')
with open('/opt/ml/code/datasets/外观机数据/防爆阀/分割/dict.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        print(line)
        labels.append(line.lower())


for i, name in enumerate(file_names):
    # load json
    json_name = os.path.join(dataset_path, name + '.json')
    print(json_name)

    with open(json_name, 'r') as f:
        coco_data = json.load(f)

    instances = []
    for x in coco_data['shapes']:
        # print(labels)
        if x['label'].lower() not in labels:
            continue
        points = []
        for point in x['points']:
            points.append(point[0])
            points.append(point[1])
        instances.append([points])
    if len(instances) == 0:
        continue

    
    kk = _convert_masks(instances, coco_data['imageHeight'], coco_data['imageWidth'])
    kk2 = [i.mask[...,np.newaxis] for i in kk]
    _mask = np.concatenate(kk2, axis=-1)
    _mask = _mask.sum(axis=-1)
    save_name = os.path.join(dataset_path, name + '_mask.bmp')
    cv2.imwrite(save_name, _mask*255)


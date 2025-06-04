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

# Load JSON
dataset_path = '/opt/ml/code/xuxichen/datasets/blue_dataset/'
save_img_path = '/opt/ml/code/xuxichen/datasets/blue_data_class/image/'
save_mask_path = '/opt/ml/code/xuxichen/datasets/blue_data_class/mask/'
os.makedirs(save_img_path,exist_ok=True)
os.makedirs(save_mask_path,exist_ok=True)

_file_names = os.listdir(dataset_path)
file_names = []
for file_name in _file_names:
    file_name = file_name.split('.')
    if file_name[1] == "json":
        file_names.append(file_name[0])
print("anomaly image number: {}".format(len(file_names)))

# Load labels
labels = []
print('labels:')
with open('/opt/ml/code/datasets/外观机数据/防爆阀/分割/dict.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        print(line)
        labels.append(line)

# labels.append("Loc")
print(labels)
for i, name in enumerate(file_names):
    # Load JSON
    json_name = os.path.join(dataset_path, name + '.json')
    print(json_name)

    with open(json_name, 'r') as f:
        coco_data = json.load(f)

    for label in labels:
        instances = []
        for x in coco_data['shapes']:
            if x['label'].lower() == label.lower():
                points = []
                for point in x['points']:
                    points.append(point[0])
                    points.append(point[1])
                instances.append([points])
        if len(instances) == 0:
            print(label)
            continue

        kk = _convert_masks(instances, coco_data['imageHeight'], coco_data['imageWidth'])

        kk2 = [i.mask[..., np.newaxis] for i in kk]
        _mask = np.concatenate(kk2, axis=-1)
        _mask = _mask.sum(axis=-1)
        save_name_mask = os.path.join(save_mask_path, name + '_' + label + '_mask.bmp')
        save_name_img = os.path.join(save_img_path, name + '_' + label + '.bmp')
        
        image_name = os.path.join(dataset_path, name + '.bmp')  # Assuming BMP format for image files
        image = cv2.imread(image_name)

        cv2.imwrite(save_name_mask, _mask * 255)
        cv2.imwrite(save_name_img, image)
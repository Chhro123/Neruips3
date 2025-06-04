import os
import json
import cv2
import numpy as np
import pycocotools.mask as mask_util
from visualizer import GenericMask

def _convert_masks(masks_or_polygons, height, width):
    ret = []
    for x in masks_or_polygons:
        if isinstance(x, GenericMask):
            ret.append(x)
        else:
            ret.append(GenericMask(x, height, width))
    return ret

dataset_path = '/opt/ml/code/xuxichen/datasets/blue_dataset/'
save_path = '/opt/ml/code/xuxichen/datasets/blue_dataset_256_all/'
file_names = [fn.split('.')[0] for fn in os.listdir(dataset_path) if fn.endswith('.json')]
print("anomaly image number: {}".format(len(file_names)))

labels = []
with open('/opt/ml/code/datasets/外观机数据/防爆阀/分割/dict.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print('labels:', labels)

for name in file_names:
    print(name)
    json_name = os.path.join(dataset_path, name + '.json')
    with open(json_name, 'r') as f:
        coco_data = json.load(f)

    image_name = os.path.join(dataset_path, name + '.bmp')  # Assuming BMP format for image files
    image = cv2.imread(image_name)
    count = 0
    instances = []
    for shape in coco_data['shapes']:
        if shape['label'] not in labels:
            continue
        points = [pt for sublist in shape['points'] for pt in sublist]
        instances.append([points])

# 转化了所有mask，然后去进行遍历    
    masks = _convert_masks(instances, coco_data['imageHeight'], coco_data['imageWidth'])
    for i, mask in enumerate(masks):
        
        full_mask = mask.mask[..., np.newaxis] * 255
        shape = coco_data['shapes'][i]  # 获取当前正在处理的shape
        points = shape['points']

        # 计算x和y坐标
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        xmin, xmax = max(0, min(x_coords)), min(coco_data['imageWidth'], max(x_coords))
        ymin, ymax = max(0, min(y_coords)), min(coco_data['imageHeight'], max(y_coords))
        
        # Define the patch size and calculate center point
        patch_size = 256
        xc = min(xmax, max(xmin, (xmin + xmax) // 2))
        yc = min(ymax, max(ymin, (ymin + ymax) // 2))
        
        
        # Adjust the crop area to not exceed image boundaries
        x0 = int(max(0, xc - patch_size // 2))
        y0 = int(max(0, yc - patch_size // 2))
        x1 = int(min(coco_data['imageWidth'], xc + patch_size // 2))
        y1 = int(min(coco_data['imageHeight'], yc + patch_size // 2))
        
        cropped_mask = full_mask[y0:y1, x0:x1]
        cropped_image = image[y0:y1, x0:x1]
        
        # Save the mask and image with the corresponding label in the file name
        label = shape['label']
        mask_save_name = os.path.join(save_path, f"{name}_shape_{count}_{label}_mask.bmp")
        image_save_name = os.path.join(save_path, f"{name}_shape_{count}_{label}.bmp")
        
        cv2.imwrite(mask_save_name, cropped_mask)
        cv2.imwrite(image_save_name, cropped_image)
        count+=1

print("Processing complete.")

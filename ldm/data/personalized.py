import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
import random
from .mask_generate import generate_mask
import cv2
from utils import random_transform
import numpy as np
imagenet_templates_smallest = [
    #  '{} lan_fangbaofa'
    'Anomaly {}',
]

def get_files(folder_path):
    bmp_files_list = []  # 初始化一个空列表来存储文件路径
    for filename in os.listdir(folder_path):
        # 检查文件扩展名是否为bmp
        if filename.lower().endswith('.bmp') or filename.lower().endswith('.png') or filename.lower().endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)  # 获取文件的完整路径
            bmp_files_list.append(file_path)  # 将文件路径添加到列表中
    return bmp_files_list  # 返回包含文件路径的列表



per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

# 先训一版蓝膜
class Personalized_mvtec_encoder(Dataset):
    def __init__(self,
                 mvtec_path,
                 size=None,
                 repeats=1,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 data_enhance=False,
                 random_mask=False,
                 **kwargs
                 ):
        
        self.data_enhance=None
        if data_enhance:
            self.data_enhance=random_transform()

        self.data_root=mvtec_path

        self.data=[]
       
        self.size = size
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]


        img_path=os.path.join(self.data_root,'image/')
        mask_path=os.path.join(self.data_root,'mask/')

        img_files = get_files(img_path)
        mask_files = get_files(mask_path)


        img_files=[os.path.join(self.data_root,file_name) for file_name in img_files]
        mask_files=[os.path.join(self.data_root,file_name) for file_name in mask_files]

        for idx in range(len(img_files)):
            img_filename = img_files[idx]

            mask_filename = img_filename.split('.')[0] + '_mask.bmp'
            mask_filename = mask_filename.replace("/image/", "/mask/")

            if not os.path.exists(mask_filename):
                print("jjjjjjjjjjjjjjjjj")
                print(mask_filename)
                continue
           
            name = os.path.basename(img_filename).split('.')[0]
            label_string = name.split('_')[-1]
          
            anomaly_name = label_string


            image = Image.open(img_filename)
            mask = Image.open(mask_filename).convert("L")
            if not image.mode == "RGB":
                image = image.convert("RGB")

            image = np.array(image).astype(np.uint8)
            mask = np.array(mask).astype(np.float32)

            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            # # 为了确保图像正确，resize到256x256
            size = 256
            image = image.resize((size, size), resample=self.interpolation)
            mask = mask.resize((size, size), resample=self.interpolation)
            image = np.array(image).astype(np.float32)
            mask = np.array(mask).astype(np.float32)
            image= (image / 127.5 - 1.0).astype(np.float32)
            mask = mask / 255.0
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
    
            # self.data.append((image,mask,anomaly_name))
            self.data.append((image,mask,anomaly_name))

        self.num_images = len(self.data)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats
        else:
            self._length = 4
        self.random_mask=random_mask


    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        idx=idx%self.num_images
        example = {}
        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        text = random.choice(imagenet_templates_smallest).format(placeholder_string)
        image=self.data[idx][0]
        if self.random_mask:
            mask=generate_mask(256)
        else:
            mask=self.data[idx][1]
        example["caption"] = text
        example["image"] = image
        example["mask"] = mask
        example["name"]=self.data[idx][2]
    
        return example


class MvtecDataset (Personalized_mvtec_encoder):
    def __init__(self,
                 mvtec_path,
                 size=None,
                 repeats=1,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 data_enhance=False,
                 random_mask=False,
                 **kwargs
                 ):
        
        self.data_enhance=None
        if data_enhance:
            self.data_enhance=random_transform()

        self.data_root=mvtec_path

        self.data=[]
       
        self.size = size
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]


        mask_path=os.path.join(self.data_root,'defect/')
        mask_files = get_files(mask_path)


        # img_files=[os.path.join(self.data_root,file_name) for file_name in img_files]
        # mask_files=[os.path.join(self.data_root,file_name) for file_name in mask_files]

        for idx in range(len(mask_files)):
            mask_filename = mask_files[idx]

            img_filename = mask_filename.split('.')[0][0:-5] +".png"
            img_filename = img_filename.replace("/defect/", "/good/")

            if not os.path.exists(img_filename):
                print("jjjjjjjjjjjjjjjjj")
                print(mask_filename)
                continue
           
            parts = img_filename.split(os.sep)
            # 获取所需的部分
            object = parts[7]  # 'bottle'
            broken = os.path.basename(img_filename).split('.')[0][0:-4]
            # 用 + 连接
            anomaly_name = object + '+' + broken
            # print(anomaly_name)

            image = Image.open(img_filename)
            mask = Image.open(mask_filename).convert("L")
            if not image.mode == "RGB":
                image = image.convert("RGB")

            image = np.array(image).astype(np.uint8)
            mask = np.array(mask).astype(np.float32)

            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            # # 为了确保图像正确，resize到256x256
            size = 256
            image = image.resize((size, size), resample=self.interpolation)
            mask = mask.resize((size, size), resample=self.interpolation)
            image = np.array(image).astype(np.float32)
            mask = np.array(mask).astype(np.float32)
            image= (image / 127.5 - 1.0).astype(np.float32)
            mask = mask / 255.0
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
    
            # self.data.append((image,mask,anomaly_name))
            self.data.append((image,mask,anomaly_name))

        self.num_images = len(self.data)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats
        else:
            self._length = 4
        self.random_mask=random_mask


    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        idx=idx%self.num_images
        example = {}
        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        text = random.choice(imagenet_templates_smallest).format(placeholder_string)
        image=self.data[idx][0]
        if self.random_mask:
            mask=generate_mask(256)
        else:
            mask=self.data[idx][1]
        example["caption"] = text
        example["image"] = image
        example["mask"] = mask
        example["name"]=self.data[idx][2]
    
        return example

class MvtecDataset_singel_anomaly (Personalized_mvtec_encoder):
    def __init__(self,
                 mvtec_path,
                 size=None,
                 repeats=1,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 data_enhance=False,
                 random_mask=False,
                 **kwargs
                 ):
        
        self.data_enhance=None
        if data_enhance:
            self.data_enhance=random_transform()

        self.data_root=mvtec_path

        self.data=[]
       
        self.size = size
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]


        mask_path = self.data_root
        mask_files = get_files(mask_path)



        for idx in range(len(mask_files)):
            mask_filename = mask_files[idx]

            img_filename = mask_filename.split('.')[0][0:-5] +".png"
            img_filename = img_filename.replace("/ground_truth/", "/test/")

            if not os.path.exists(img_filename):
                print("jjjjjjjjjjjjjjjjj")
                print(mask_filename)
                continue
           
            parts = img_filename.split(os.sep)
            # 获取所需的部分
            object = parts[7]  # 'bottle'
            broken = parts[9]
            # 用 + 连接
            anomaly_name = object + '+' + broken
            # print(anomaly_name)

            image = Image.open(img_filename)
            mask = Image.open(mask_filename).convert("L")
            if not image.mode == "RGB":
                image = image.convert("RGB")

            image = np.array(image).astype(np.uint8)
            mask = np.array(mask).astype(np.float32)

            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            # # 为了确保图像正确，resize到256x256
            size = 256
            image = image.resize((size, size), resample=self.interpolation)
            mask = mask.resize((size, size), resample=self.interpolation)
            image = np.array(image).astype(np.float32)
            mask = np.array(mask).astype(np.float32)
            image= (image / 127.5 - 1.0).astype(np.float32)
            mask = mask / 255.0
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
    
            # self.data.append((image,mask,anomaly_name))
            self.data.append((image,mask,anomaly_name))

        self.num_images = len(self.data)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats
        else:
            self._length = 4
        self.random_mask=random_mask


    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        idx=idx%self.num_images
        example = {}
        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        text = random.choice(imagenet_templates_smallest).format(placeholder_string)
        image=self.data[idx][0]
        if self.random_mask:
            mask=generate_mask(256)
        else:
            mask=self.data[idx][1]
        example["caption"] = text
        example["image"] = image
        example["mask"] = mask
        example["name"]=self.data[idx][2]
    
        return example




class MvtecDataset_singel_anomaly_validation (Personalized_mvtec_encoder):
    def __init__(self,
                 mvtec_path,
                 size=None,
                 repeats=1,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 data_enhance=False,
                 random_mask=False,
                 **kwargs
                 ):
        
        self.data_enhance=None
        if data_enhance:
            self.data_enhance=random_transform()

        self.data_root=mvtec_path

        self.data=[]
       
        self.size = size
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]


        # mask_path=os.path.join(self.data_root,'ground_truth/')
        mask_path = self.data_root
        mask_files = get_files(mask_path)


        for idx in range(len(mask_files)):
            mask_filename = mask_files[idx]

            img_filename = mask_filename.split('.')[0][0:-5] +".png"
            img_filename = img_filename.replace("/ground_truth/", "/train/")

           
            parts = img_filename.split(os.sep)
            # 获取所需的部分
            object1 = parts[7]  # 'bottle'
            broken = parts[9]
            # 用 + 连接
            anomaly_name = object1 + '+' + broken

            # 这里重新修改img的路径
            parts[9] = "good"
            # 重新拼接路径
            img_filename = os.sep.join(parts)

            if not os.path.exists(img_filename):
                print("jjjjjjjjjjjjjjjjj")
                print(mask_filename)
                continue

            
            image = Image.open(img_filename)
            mask = Image.open(mask_filename).convert("L")
            if not image.mode == "RGB":
                image = image.convert("RGB")

            image = np.array(image).astype(np.uint8)
            mask = np.array(mask).astype(np.float32)

            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            # # 为了确保图像正确，resize到256x256
            size = 256
            image = image.resize((size, size), resample=self.interpolation)
            mask = mask.resize((size, size), resample=self.interpolation)
            image = np.array(image).astype(np.float32)
            mask = np.array(mask).astype(np.float32)
            image= (image / 127.5 - 1.0).astype(np.float32)
            mask = mask / 255.0
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
    
            # self.data.append((image,mask,anomaly_name))
            self.data.append((image,mask,anomaly_name))

        self.num_images = len(self.data)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats
        else:
            self._length = 4
        self.random_mask=random_mask


    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        idx=idx%self.num_images
        example = {}
        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        text = random.choice(imagenet_templates_smallest).format(placeholder_string)
        image=self.data[idx][0]
        if self.random_mask:
            mask=generate_mask(256)
        else:
            mask=self.data[idx][1]
        example["caption"] = text
        example["image"] = image
        example["mask"] = mask
        example["name"]=self.data[idx][2]
    
        return example
    
    
class Mvtec_generation_dataset (Personalized_mvtec_encoder):
    def __init__(self,
                 mvtec_path,
                 mask_path,
                 sample_name,
                 anomaly_name,
                 size=256,
                 repeats=1,
                 interpolation="bicubic",
                 flip_p=0.5,
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 random_mask=False,
                 **kwargs
                 ):
        
       

        self.data=[]
        self.size = size
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]

        self.data_root = mvtec_path
        self.mask_path= mask_path
        self.img_path=self.data_root


        img_files = get_files(self.img_path)
        mask_files = get_files(self.mask_path)


        self.img_files = img_files
        self.mask_files = mask_files

        self.num_images = len(self.data)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats
        else:
            self._length = 4
        self.random_mask=random_mask


    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        # idx=idx%self.num_images
        example = {}
        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        text = random.choice(imagenet_templates_smallest).format(placeholder_string)
        img_filename = self.img_files[random.randint(0,len(self.img_files)-1)]
        mask_filename = self.mask_files[random.randint(0,len(self.mask_files)-1)]

        
        parts = mask_filename.split(os.sep)
        # 获取所需的部分
        object1 = parts[7]  # 'bottle'
        broken = parts[9]
        # 用 + 连接
        anomaly_name = object1 + '+' + broken
    
        image = Image.open(img_filename)
        mask = Image.open(mask_filename).convert("L")
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.float32)

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        # # 为了确保图像正确，resize到256x256
        size = 256
        image = image.resize((size, size), resample=self.interpolation)
        mask = mask.resize((size, size), resample=self.interpolation)
        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        image= (image / 127.5 - 1.0).astype(np.float32)
        mask = mask / 255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1


        example["caption"] = text
        example["image"] = image
        example["mask"] = mask
        example["name"]= anomaly_name
    
        return example
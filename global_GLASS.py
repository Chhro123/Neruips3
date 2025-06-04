import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from enum import Enum
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# 假设 perlin_mask 函数已经定义并导入
from perlin import perlin_mask

# 定义类别名称及相关常量
_CLASSNAMES = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class MVTecDataset(Dataset):
    """
    修改后的 MVTec Dataset，返回异常图像及对应的两种掩码（mask_s和mask_l）。
    为生成图像级别异常，本例在 TRAIN 模式下合成异常图像，并返回全分辨率的异常掩码 mask_l。
    """
    def __init__(
            self,
            source,
            anomaly_source_path='/root/dataset/dtd/images',
            dataset_name='mvtec',
            classname='leather',
            resize=288,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            rotate_degrees=0,
            translate=0,
            brightness_factor=0,
            contrast_factor=0,
            saturation_factor=0,
            gray_p=0,
            h_flip_p=0,
            v_flip_p=0,
            distribution=0,
            mean=0.5,
            std=0.1,
            fg=0,
            rand_aug=1,
            downsampling=8,
            scale=0,
            batch_size=8,
            **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.batch_size = batch_size
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.fg = fg
        self.rand_aug = rand_aug
        self.downsampling = downsampling
        self.resize = resize if self.distribution != 1 else [resize, resize]
        self.imgsize = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)
        self.classname = classname
        self.dataset_name = dataset_name

        if self.distribution != 1 and (self.classname == 'toothbrush' or self.classname == 'wood'):
            self.resize = round(self.imgsize * 292 / 256)

        xlsx_path = './datasets/excel/' + self.dataset_name + '_distribution.xlsx'
        if self.fg == 2:  # choose by file
            try:
                df = pd.read_excel(xlsx_path)
                self.class_fg = df.loc[df['Class'] == self.dataset_name + '_' + classname, 'Foreground'].values[0]
            except:
                self.class_fg = 1
        elif self.fg == 1:  # with foreground mask
            self.class_fg = 1
        else:  # without foreground mask
            self.class_fg = 0

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        self.anomaly_source_paths = sorted(
            glob.glob(anomaly_source_path + "/*/*.jpg") +
            []  
        )

        self.transform_img = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0 - scale, 1.0 + scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
        ])

    def rand_augmenter(self):
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)
        transform_aug = transforms.Compose([
            transforms.Resize(self.resize),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform_aug

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        # 默认掩码前景
        mask_fg = torch.tensor([1])
        if self.split == DatasetSplit.TRAIN:
            aug_texture = Image.open(np.random.choice(self.anomaly_source_paths)).convert("RGB")
            if self.rand_aug:
                transform_aug = self.rand_augmenter()
                aug_texture = transform_aug(aug_texture)
            else:
                aug_texture = self.transform_img(aug_texture)

            if self.class_fg:
                fgmask_path = "/Data/xxc/GLASS/All_fg_mask/MVTec AD/" + 'fg_mask/' + classname + '/' + os.path.split(image_path)[-1]
                mask_fg_img = Image.open(fgmask_path)
                mask_fg = torch.ceil(self.transform_mask(mask_fg_img)[0])

            mask_all = perlin_mask(image.shape, self.imgsize // self.downsampling, 0, 6, mask_fg, 1)
            mask_s = torch.from_numpy(mask_all[0])
            mask_l = torch.from_numpy(mask_all[1])

            beta = np.random.normal(loc=self.mean, scale=self.std)
            beta = np.clip(beta, 0.2, 0.8)
            aug_image = image * (1 - mask_l) + (1 - beta) * aug_texture * mask_l + beta * image * mask_l
        else:
            aug_image = image
            mask_s = torch.zeros([1, image.size()[1], image.size()[2]])
            mask_l = mask_s.clone()

        # 对于测试集，加载真实异常掩码（若存在）
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask_gt = Image.open(mask_path).convert('L')
            mask_gt = self.transform_mask(mask_gt)
        else:
            mask_gt = torch.zeros([1, image.size()[1], image.size()[2]])

        return {
            "image": image,
            "aug": aug_image,      # 异常图像
            "mask_s": mask_s,      # 下采样后的异常掩码
            "mask_l": mask_l,      # 全分辨率异常掩码
            "mask_gt": mask_gt,
            "is_anomaly": int(anomaly != "good"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        classpath = os.path.join(self.source, self.classname, self.split.value)
        maskpath = os.path.join(self.source, self.classname, "ground_truth")
        anomaly_types = os.listdir(classpath)

        imgpaths_per_class[self.classname] = {}
        maskpaths_per_class[self.classname] = {}

        for anomaly in anomaly_types:
            anomaly_path = os.path.join(classpath, anomaly)
            anomaly_files = sorted(os.listdir(anomaly_path))
            imgpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_path, x) for x in anomaly_files]

            if self.split == DatasetSplit.TEST and anomaly != "good":
                anomaly_mask_path = os.path.join(maskpath, anomaly)
                anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                maskpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files]
            else:
                maskpaths_per_class[self.classname]["good"] = None

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


def unnormalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp(0, 1)


def save_anomaly_dataset(dataset, classnames=_CLASSNAMES, num_samples_per_class=8000, out_img_dir="./output_anomalies/images", out_mask_dir="./output_anomalies/masks"):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    
    count = 0
    idx = 0
    total = len(dataset)
    
    for classname in classnames:
        dataset.classname = classname  # 动态修改类别
        
        while count < num_samples_per_class:
            sample = dataset[idx % total]
            idx += 1
            # 使用全分辨率异常掩码 mask_l 判断前景区域是否存在
            mask = sample["mask_l"]  # shape: [H, W]（假定为单通道）
            if torch.sum(mask) == 0:
                continue  # 若无前景，则跳过
            anomaly_tensor = sample["aug"]
            anomaly_tensor = unnormalize(anomaly_tensor)
            anomaly_img = transforms.ToPILImage()(anomaly_tensor)
            img_save_path = os.path.join(out_img_dir, f"{count:04d}.bmp")
            anomaly_img.save(img_save_path, format="BMP")

            mask_np = (mask.numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np, mode="L")
            mask_save_path = os.path.join(out_mask_dir, f"{count:04d}.png")
            mask_img.save(mask_save_path, format="PNG")
            count += 1
            if count % 100 == 0:
                print(f"已保存 {count} 张样本（类别: {classname}）")

    print(f"共保存 {count} 张异常图像及对应异常掩码")

# 动态初始化 MVTecDataset
def initialize_and_save_images(mvtec_root, anomaly_texture_root, classnames=_CLASSNAMES, num_samples_per_class=5000, batch_size=16):
    for classname in classnames:
        print(f"正在处理类别：{classname}")
        
        # 初始化数据集
        dataset = MVTecDataset(
            source=mvtec_root,
            anomaly_source_path=anomaly_texture_root,
            classname=classname,  # 每次循环传递不同的类别名
            split=DatasetSplit.TRAIN,
            resize=256,  
            imagesize=256,
            rotate_degrees=0,
            translate=0,
            brightness_factor=0,
            contrast_factor=0,
            saturation_factor=0,
            gray_p=0,
            h_flip_p=0,
            v_flip_p=0,
            distribution=0,
            mean=0.5,
            std=0.1,
            fg=1,
            rand_aug=1,
            downsampling=8,
            scale=0,
            batch_size=batch_size
        )
        
        # 调用保存图像和掩码的函数
        save_anomaly_dataset(dataset, classnames=[classname], num_samples_per_class=num_samples_per_class,
                             out_img_dir=f"./output_anomalies/{classname}/images",
                             out_mask_dir=f"./output_anomalies/{classname}/mask")


if __name__ == "__main__":
    # 根据实际情况设置 MVTec 数据集路径及其它参数
    mvtec_root = "../MVTEC_AD"  # 修改为你的数据集路径
    anomaly_texture_root = "/Data/xxc/GLASS/dtd/images"  # 修改为异常纹理数据路径

    # 调用初始化函数进行异常图像和掩码保存
    initialize_and_save_images(mvtec_root, anomaly_texture_root, classnames=_CLASSNAMES, num_samples_per_class=4200, batch_size=8)

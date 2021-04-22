import os
import os.path
import torch
import torch.utils.data as data
from PIL import Image
from PIL import ImageEnhance
import random


def make_dataset(root):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowImages')) if f.endswith('.jpg')]
    return [
        (os.path.join(root, 'ShadowImages', img_name + '.jpg'), os.path.join(root, 'ShadowMasks', img_name + '.png'),\
        os.path.join(root, 'fuse_dst1', img_name + '.png'),os.path.join(root, 'fuse_dst2', img_name + '.png'))
        for img_name in img_list]


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, dst1_path, dst2_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        dst1 = Image.open(dst1_path)
        dst2 = Image.open(dst2_path)
        if self.joint_transform is not None:
            img, target, dst1, dst2 = self.joint_transform(img, target, dst1, dst2)
        if self.transform is not None:
            # img = random.uniform(1.0,1.2)*img
            # img = torch.clamp(img, 0, 255)
            # enh_bri = ImageEnhance.Brightness(img)
            # img = enh_bri.enhance(random.uniform(1.0,1.2))
            # img = torch.clamp(img, 0, 255)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            dst1 = self.target_transform(dst1)
            dst2 = self.target_transform(dst2)
        return img, target, dst1, dst2

    def __len__(self):
        return len(self.imgs)

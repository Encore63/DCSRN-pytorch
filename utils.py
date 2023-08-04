import os
import PIL
import time
import torch
import shutil
import random
import cv2 as cv
import numpy as np
import torch.fft as fft
import SimpleITK as sitk

from tqdm import tqdm
from PIL import Image
from torchvision import transforms


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def PSNR(image_1, image_2):
    mse = torch.mean((image_1 - image_2) ** 2)
    return 20. * torch.log10(255 / torch.sqrt(mse))


def calc_psnr(patch_1, patch_2, patch_size=64, image_size=256):
    image_1 = fft_postprocess(patch_1, patch_size, image_size)
    image_2 = fft_postprocess(patch_2, patch_size, image_size)
    psnr, total = 0., (image_1.shape[0] + image_2.shape[0]) // 2
    for dim in range(total):
        psnr += PSNR(image_1[dim, :, :], image_2[dim, :, :])
    return psnr / total


def gaussian_high_freq_filter(fft_data: torch.Tensor, D) -> torch.Tensor:
    """
    高斯滤波器
    :param fft_data: 待滤波数据
    :param D: 高斯滤波截止频率
    :return: 张量
    """
    h, w = fft_data.shape
    x, y = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    center = (int((h - 1) / 2), int((w - 1) / 2))

    distance_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = torch.exp(- distance_square / (2 * D ** 2))

    return template * fft_data


def circle_low_freq_filter(fft_data: torch.Tensor, radius_ratio) -> torch.Tensor:
    """
    圆形低频滤波器
    :param fft_data: 待滤波数据
    :param radius_ratio: 滤波器半径
    :return: 张量
    """
    assert 1 < len(fft_data.shape) < 4
    fft_data = np.array(fft_data)
    template = np.zeros(fft_data.shape, dtype=np.uint8)
    c_row, c_col = int(fft_data.shape[0] / 2), int(fft_data.shape[1] / 2)
    radius = int(radius_ratio * fft_data.shape[0] / 2)
    if len(fft_data.shape) == 3:
        cv.circle(template, (c_row, c_col), radius, (1, 1, 1), -1)
    else:
        cv.circle(template, (c_row, c_col), radius, 1, -1)
    template = torch.from_numpy(template)
    return template * fft_data


def circle_high_freq_filter(fft_data: torch.Tensor, radius_ratio) -> torch.Tensor:
    """
    圆形高频滤波器
    :param fft_data: 待滤波数据
    :param radius_ratio: 滤波器半径
    :return: 张量
    """
    assert 1 < len(fft_data.shape) < 4
    fft_data = np.array(fft_data)
    template = np.ones(fft_data.shape, dtype=np.uint8)
    c_row, c_col = int(fft_data.shape[0] / 2), int(fft_data.shape[1] / 2)
    radius = int(radius_ratio * fft_data.shape[0] / 2)
    if len(fft_data.shape) == 3:
        cv.circle(template, (c_row, c_col), radius, (0, 0, 0), -1)
    else:
        cv.circle(template, (c_row, c_col), radius, 0, -1)
    template = torch.from_numpy(template)
    return template * fft_data


def image_blur(data: torch.Tensor, radius: float = 0.75) -> torch.Tensor:
    assert len(data.shape) <= 3
    if len(data.shape) == 3:
        data = data.squeeze(0)
    fft_data = fft.fftn(data)
    blur_data = circle_high_freq_filter(fft_data, radius_ratio=radius)
    blur_data = torch.abs(fft.ifftn(blur_data)).unsqueeze(0)
    return blur_data


def sequential_patch(data: torch.Tensor, patch_size=64) -> torch.Tensor:
    """
    MRI图像顺序分块
    :param data: 原始图像数据
    :param patch_size: 块大小
    :return: 张量
    """
    patches = []
    for sub_img in range(data.shape[0]):
        for idx_x in range(data.shape[1] // patch_size):
            idx_x *= patch_size
            for idx_y in range(data.shape[2] // patch_size):
                idx_y *= patch_size
                patch = torch.zeros((patch_size, patch_size))
                patch[:, :] = data[sub_img, idx_x:idx_x + patch_size, idx_y:idx_y + patch_size]
                patches.append(patch.unsqueeze(0))
    patches = torch.cat(patches, dim=0)
    return patches


def sequential_merge(data: torch.Tensor, patch_size=64, image_size=256) -> torch.Tensor:
    """
    MRI分块图像顺序归并
    :param data: 分块图像数据
    :param patch_size: 块大小
    :param image_size: 原始图像大小
    :return: 张量
    """
    idx_z = 0
    image_cube = []
    image = torch.zeros((image_size, image_size))
    data = data.reshape((-1, patch_size, patch_size))
    scale = image_size // patch_size
    while idx_z != data.shape[0]:
        if idx_z % (scale ** 2) == 0:
            image = torch.zeros((image_size, image_size))
        for idx_x in range(scale):
            idx_x *= patch_size
            for idx_y in range(scale):
                idx_y *= patch_size
                image[idx_x:idx_x + patch_size, idx_y:idx_y + patch_size] = data[idx_z, :, :]
                idx_z += 1
                if idx_x + patch_size == image_size and idx_y + patch_size == image_size:
                    image_cube.append(image.unsqueeze(0))
    image_cube = torch.cat(image_cube, dim=0)
    return image_cube


def random_patch(data: torch.Tensor, patch_size=64) -> torch.Tensor:
    """
    MRI图像随机分块
    :param data: 原始图像数据
    :param patch_size: 块大小
    :return:张量
    """
    assert len(data.shape) == 3
    np.random.seed(int(time.time()))
    x_shape = data.shape[0]
    y_shape = data.shape[1]
    z_shape = data.shape[2]
    x_idx = np.random.randint(0, x_shape - patch_size)
    y_idx = np.random.randint(0, y_shape - patch_size)
    z_idx = np.random.randint(0, z_shape - patch_size)

    patch = data[x_idx: x_idx + patch_size, y_idx: y_idx + patch_size, z_idx: z_idx + patch_size]

    return patch


def get_random_patches(data: torch.Tensor, patch_size=64):
    assert data.shape[1] >= patch_size
    data_patches = []
    for dim in range(data.shape[0]):
        data_patch = random_patch(data[dim], patch_size)
        data_patches.append(data_patch)
    data_patches = torch.cat(data_patches, dim=0).reshape((-1, patch_size, patch_size, patch_size))
    return data_patches


def get_sequential_patches(data: torch.Tensor, patch_size=64):
    assert data.shape[1] >= patch_size
    data_patches = []
    for dim in range(data.shape[0]):
        data_patch = sequential_patch(data[dim], patch_size)
        for idx in range(data_patch.shape[0] // patch_size):
            idx *= patch_size
            data_patches.append(data_patch[idx:idx + patch_size, :, :].unsqueeze(0))
    data_patches = torch.cat(data_patches, dim=0).reshape((-1, patch_size, patch_size, patch_size))
    return data_patches


def dataset_spilt(_src_path: str, dataset: str, train_ratio: float, val_ratio: float) -> None:
    """
    :param _src_path: 数据根目录
    :param dataset: 数据集名称
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    """
    assert train_ratio + val_ratio < 1.0

    files = os.listdir(os.path.join(_src_path, dataset))
    train_path = os.path.join(_src_path, 'train')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    val_path = os.path.join(_src_path, 'val')
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    test_path = os.path.join(_src_path, 'test')
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    random.seed(time.time())
    _src_path = os.path.join(_src_path, dataset)
    for file in files:
        src = os.path.join(_src_path, file)
        rand = random.random()
        if rand <= train_ratio:
            dst = os.path.join(train_path, file)
        elif train_ratio < rand <= train_ratio + val_ratio:
            dst = os.path.join(val_path, file)
        else:
            dst = os.path.join(test_path, file)
        shutil.copyfile(src, dst)


def bicubic_preprocess(_src_path: str, _scale: float, _step: int, transform: transforms.ToTensor) -> tuple:
    """
    :param _src_path: 数据集路径
    :param _scale: 图像缩放倍率
    :param _step: 图像读取步长
    :param transform: 张量转换器
    :return: 包含数据和标签的元组
    """
    files = os.listdir(_src_path)
    images, targets = [], []
    paths = [os.path.join(_src_path, file) for file in files]
    loop = tqdm(enumerate(paths), total=len(paths), position=0)
    for idx, path in loop:
        image: sitk.Image = sitk.ReadImage(path)
        image: np.ndarray = sitk.GetArrayFromImage(image)
        for i in range(0, image.shape[0], _step):
            im = image[i, :, :]
            im: Image.Image = Image.fromarray(im)
            # 高分辨率图像作为标签
            targets.append(torch.tensor(transform(im), dtype=torch.float32))
            # 使用双三次插值对原始图像进行降采样
            im = im.resize(size=(int(im.height / _scale), int(im.width / _scale)), resample=PIL.Image.BICUBIC)
            im = im.resize(size=(int(im.height * _scale), int(im.width * _scale)), resample=PIL.Image.BICUBIC)
            # 降采样后的低分辨率图像作为数据
            images.append(torch.tensor(transform(im), dtype=torch.float32))
        loop.set_description('File Preprocess')
        loop.set_postfix(file='{}'.format(path.split('\\')[-1]))
    return images, targets


def fft_preprocess(_src_path: str, _dst_path: str, _step: int, _radius=0.75, patch_size=64):
    """
    利用快速傅里叶变换将MRI图像映射到K空间，将75%的高频部分归零实现图像的模糊处理
    """
    files = os.listdir(_src_path)
    lr_imgs, hr_imgs = [], []
    lr_patches, hr_patches = torch.zeros((1, 1, 64, 64, 64)), torch.zeros((1, 1, 64, 64, 64))
    paths = [os.path.join(_src_path, file) for file in files]
    if 'train' in _src_path:
        paths = paths[:int(len(paths) * 0.1)]
    loop = tqdm(enumerate(paths), total=len(paths), position=0)
    for idx, path in loop:
        lr_cat, hr_cat = [], []
        image: sitk.Image = sitk.ReadImage(path)
        image: np.ndarray = sitk.GetArrayFromImage(image)
        image: torch.Tensor = torch.from_numpy(image)
        for i in range(0, 128, _step):
            hr_im = image[i, :, :]
            hr_cat.append(hr_im.unsqueeze(0).to(torch.float32))
            lr_im = image_blur(hr_im, radius=_radius)
            lr_cat.append(lr_im.to(torch.float32))
        lr_imgs.append(torch.cat(lr_cat, dim=0).unsqueeze(0))
        hr_imgs.append(torch.cat(hr_cat, dim=0).unsqueeze(0))
        loop.set_description('File Preprocess')
        loop.set_postfix(file='{}'.format(path.split('\\')[-1]))
    lr_imgs, hr_imgs = torch.cat(lr_imgs, dim=0), torch.cat(hr_imgs, dim=0)
    lr_patches, hr_patches = get_sequential_patches(lr_imgs, patch_size), get_sequential_patches(hr_imgs, patch_size)
    lr_patches = np.array(lr_patches.reshape((-1, 1, 64, 64, 64)), dtype=np.float)
    hr_patches = np.array(hr_patches.reshape((-1, 1, 64, 64, 64)), dtype=np.float)
    np.save(os.path.join(_dst_path, 'lr_data.npy'), lr_patches)
    np.save(os.path.join(_dst_path, 'hr_data.npy'), hr_patches)


def fft_postprocess(data_patch: torch.Tensor, patch_size=64, image_size=256) -> torch.Tensor:
    """
    MRI图像数据后处理
    """
    image_cube: torch.Tensor = sequential_merge(data_patch, patch_size, image_size)
    return image_cube


def load_images(_src_path: str) -> tuple:
    lr_data = np.load(os.path.join(_src_path, 'lr_data.npy'))
    hr_data = np.load(os.path.join(_src_path, 'hr_data.npy'))
    return lr_data, hr_data


if __name__ == '__main__':
    # img_path = r'F:\StudyFiles\PyCharm\computer_vision\DCSRN-pytorch-master\data\HR_IMG.png'
    # img: Image.Image = Image.open(img_path)
    # img.show()
    # to_tensor = transforms.ToTensor()
    # to_image = transforms.ToPILImage()
    # t_img: torch.Tensor = to_tensor(img).squeeze(0)
    # lf_img = image_blur(t_img, radius=0.75)
    # img: Image.Image = to_image(lf_img)
    # img.show()

    x = torch.randn((1, 128, 256, 256))
    y = get_sequential_patches(x)
    z = sequential_merge(y)
    print(y.shape, z.shape, sep=' ')
    print(x == z)

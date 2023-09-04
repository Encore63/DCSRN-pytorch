import random

import torch
import utils
import argparse
import matplotlib.pyplot as plt

from dcsrn import DCSRN
from datasets import IXIDataset
from torchvision.transforms import ToPILImage


parser = argparse.ArgumentParser()
parser.add_argument('--weight-file', type=str, default=r'./pretrained_model/x0.95/epoch_106.pth')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--patch-size', type=int, default=64)
parser.add_argument('--image-size', type=int, default=256)
args = parser.parse_args()

dataset = IXIDataset(src_path=r'./data', mode='test')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DCSRN(channels=1).to(device)
state_dict = torch.load(args.weight_file)
model.load_state_dict(state_dict)
model.eval()

fig_lr: plt.Figure = plt.figure(figsize=(8, 8))
fig_lr.canvas.manager.set_window_title('lr image')
fig_hr: plt.Figure = plt.figure(figsize=(8, 8))
fig_hr.canvas.manager.set_window_title('hr image')
fig_sr: plt.Figure = plt.figure(figsize=(8, 8))
fig_sr.canvas.manager.set_window_title('sr image')

sampler = random.randint(0, len(dataset))
lr_image, hr_image = dataset[sampler]
lr_image = torch.from_numpy(lr_image)
hr_image = torch.from_numpy(hr_image)
lr_image, hr_image = lr_image.unsqueeze(0), hr_image.unsqueeze(0)
slices = 1

lr_image = lr_image.to(device)
hr_image = hr_image.to(device)
lr_image = lr_image.to(torch.float32)
hr_image = hr_image.to(torch.float32)

with torch.no_grad():
    sr_image = model(lr_image)

lr_image = utils.fft_postprocess(lr_image, args.patch_size, args.image_size)
hr_image = utils.fft_postprocess(hr_image, args.patch_size, args.image_size)
sr_image = utils.fft_postprocess(sr_image, args.patch_size, args.image_size)

avg_psnr = utils.calc_psnr(hr_image, sr_image, args.patch_size, args.image_size)
print('Average PSNR: {}'.format(avg_psnr))
print('Sample index: {}'.format(sampler))

for index in range(sr_image.shape[0]):
    if index % 1 == 0:
        axe_lr: plt.Axes = fig_lr.add_subplot(2, 2, slices)
        axe_hr: plt.Axes = fig_hr.add_subplot(2, 2, slices)
        axe_sr: plt.Axes = fig_sr.add_subplot(2, 2, slices)

        print('PSNR: {}'.format(utils.PSNR(hr_image[index], sr_image[index])))
        axe_lr.imshow(utils.tensor2image(lr_image[index]), cmap='gray')
        axe_hr.imshow(utils.tensor2image(hr_image[index]), cmap='gray')
        axe_sr.imshow(utils.tensor2image(sr_image[index]), cmap='gray')
        axe_lr.axis('off')
        axe_hr.axis('off')
        axe_sr.axis('off')

        slices += 1

plt.show()

import utils
import torch
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
from dcsrn import DCSRN
from datasets import IXIDataset
from torchvision.transforms import ToPILImage
from torch.utils.data.dataloader import DataLoader

# prepare arguments
parser = argparse.ArgumentParser()
parser.add_argument('--weight-file', type=str, default=r'./pretrained_model/x0.95/epoch_106.pth')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--patch-size', type=int, default=64)
parser.add_argument('--image-size', type=int, default=256)
args = parser.parse_args()

test_dataset = IXIDataset(src_path=r'./data', mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

# initialize model and load weight data from pretrained model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DCSRN(channels=1).to(device)
state_dict = torch.load(args.weight_file)
model.load_state_dict(state_dict)
model.eval()

index = -1
trans = ToPILImage()
lr_im, hr_im, sr_im = None, None, None
fig_lr: plt.Figure = plt.figure(figsize=(8, 8))
fig_lr.canvas.manager.set_window_title('lr image')
fig_hr: plt.Figure = plt.figure(figsize=(8, 8))
fig_hr.canvas.manager.set_window_title('hr image')
fig_sr: plt.Figure = plt.figure(figsize=(8, 8))
fig_sr.canvas.manager.set_window_title('sr image')

global_psnr = 0.
test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader), position=0)
for _, data in test_loop:
    test_psnr = utils.AverageMeter()
    inputs, labels = data[0].to(device), data[1].to(device)
    inputs, labels = inputs.to(torch.float32), labels.to(torch.float32)

    with torch.no_grad():
        outputs = model(inputs)

    avg_psnr = utils.calc_psnr(labels, outputs, args.patch_size, args.image_size)
    test_psnr.update(avg_psnr, _)
    global_psnr += avg_psnr

    test_loop.set_description('test')
    test_loop.set_postfix(PSNR=f'{avg_psnr:.2f}')

    lr_images = utils.fft_postprocess(inputs, args.patch_size, args.image_size)
    hr_images = utils.fft_postprocess(labels, args.patch_size, args.image_size)
    sr_images = utils.fft_postprocess(outputs, args.patch_size, args.image_size)

    # lr_im = lr_images[index].unsqueeze(0)
    # hr_im = hr_images[index].unsqueeze(0)
    # sr_im = sr_images[index].unsqueeze(0)

print('Average PSNR: {}'.format(global_psnr / len(test_dataloader)))

# axe_lr: plt.Axes = fig_lr.add_subplot(111)
# axe_hr: plt.Axes = fig_hr.add_subplot(111)
# axe_sr: plt.Axes = fig_sr.add_subplot(111)
#
# axe_lr.imshow(trans(lr_im), cmap='gray')
# axe_lr.axis('off')
# axe_hr.imshow(trans(hr_im), cmap='gray')
# axe_hr.axis('off')
# axe_sr.imshow(trans(sr_im), cmap='gray')
# axe_sr.axis('off')
#
# plt.show()

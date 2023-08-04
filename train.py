import os
import torch
import utils
import argparse

from tqdm import tqdm
from dcsrn import DCSRN
from utils import AverageMeter
from datasets import IXIDataset
from torch.utils.data.dataloader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--outputs-dir', type=str, default=r'./pretrained_model')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--num-epochs', type=int, default=50)
parser.add_argument('--growth-rate', type=int, default=24)
parser.add_argument('--patch-size', type=int, default=64)
parser.add_argument('--image-size', type=int, default=256)
parser.add_argument('--filter-radius', type=float, default=0.75)
parser.add_argument('--is-pretrained', type=bool, default=False)
args = parser.parse_args()

args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.filter_radius))
if not os.path.exists(args.outputs_dir):
    os.makedirs(args.outputs_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = DCSRN(channels=1, growth_rate=args.growth_rate).to(device)
if args.is_pretrained:
    state_dict = torch.load(r'./pretrained_model/')
    model.load_state_dict(state_dict)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_dataset = IXIDataset(src_path='./data', mode='train')
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
eval_dataset = IXIDataset(src_path='./data', mode='val')
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

for epoch in range(args.epochs):
    model.train()
    epoch_losses = AverageMeter()

    train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=0)

    for _, data in train_loop:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        epoch_losses.update(loss.item(), len(inputs))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loop.set_description('epoch: {0: <3d}/{1: <3d}'.format(epoch + 1, args.num_epochs))
        train_loop.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))

    model.eval()
    epoch_psnr = AverageMeter()

    eval_loop = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), position=0)
    for _, data in eval_loop:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        epoch_psnr.update(utils.calc_psnr(outputs, labels, args.patch_size, args.image_size), len(inputs))
        eval_loop.set_description('eval: {0: <3d}/{1: <3d}'.format(epoch + 1, args.num_epochs))
        eval_loop.set_postfix(PSNR=f'{epoch_psnr.avg:.2f}', loss=f'{loss:.2f}')

    torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

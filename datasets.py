import os
import utils
from typing import Tuple, List
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class IXIDataset(Dataset):
    def __init__(self, src_path: str, mode: str):
        """
        :param src_path: 数据根目录
        :param mode: 数据集用途（训练集：train，验证集：val，测试集：test）
        """
        super(IXIDataset, self).__init__()

        self.train_path = os.path.join(src_path, 'train')
        self.val_path = os.path.join(src_path, 'val')
        self.test_path = os.path.join(src_path, 'test')

        if mode == 'test':
            self.images, self.targets = utils.load_images(self.test_path)
        elif mode == 'train':
            self.images, self.targets = utils.load_images(self.train_path)
        elif mode == 'val':
            self.images, self.targets = utils.load_images(self.val_path)

    def __getitem__(self, index) -> Tuple:
        return self.images[index, :, :, :, :], self.targets[index, :, :, :, :]

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = IXIDataset(src_path=r'./data', mode='train')
    data_iter = DataLoader(dataset, batch_size=128, shuffle=False)
    for _, (image, target) in enumerate(data_iter):
        print('{0} lr_image: {1}'.format(_, image.shape))
        print('{0} hr_image: {1}'.format(_, target.shape))

import os
import utils
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, default=r'./data/train')
    parser.add_argument('--val-path', type=str, default=r'./data/val')
    parser.add_argument('--test-path', type=str, default=r'./data/test')
    parser.add_argument('--src-train-path', type=str, default=r'../SRCNN-pytorch-master/data/train')
    parser.add_argument('--src-val-path', type=str, default=r'../SRCNN-pytorch-master/data/val')
    parser.add_argument('--src-test-path', type=str, default=r'../SRCNN-pytorch-master/data/test')
    parser.add_argument('--load-step', type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.train_path):
        os.makedirs(args.train_path)
        utils.fft_preprocess(_src_path=args.src_train_path, _dst_path=args.train_path, _step=args.load_step)
    if not os.path.exists(args.val_path):
        os.makedirs(args.val_path)
        utils.fft_preprocess(_src_path=args.src_val_path, _dst_path=args.val_path, _step=args.load_step)
    if not os.path.exists(args.test_path):
        os.makedirs(args.test_path)
        utils.fft_preprocess(_src_path=args.src_test_path, _dst_path=args.test_path, _step=args.load_step)

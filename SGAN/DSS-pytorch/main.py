import argparse
import os
from dataset import get_loader
from solver import Solver


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config.train_path, config.label_path, config.img_size, config.batch_size,
                                  filename=config.train_file, num_thread=2)
        val_loader = get_loader(config.val_path, config.label_path, config.img_size, config.batch_size,
                                    filename=config.val_file, num_thread=2)
        run = 0
        config.save_fold = "/home/charis/thesis/SGAN/DSS-pytorch/results/run-5/"
        print('Start of Solver for train...')
        train = Solver(train_loader, val_loader, None, config)
        train.train()

    elif config.mode == 'test':
        test_loader = get_loader(config.test_path, config.label_path, config.img_size, config.batch_size, mode='test',
                                 filename=config.test_file, num_thread=config.num_thread)
        if not os.path.exists(config.test_fold):
            os.mkdir(config.test_fold)
        print('Start of Solver for test ...')
        test = Solver(None, None, test_loader, config)
        test.test(7200, use_crf=config.use_crf)
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    data_root = '/home/charis/kul-thesis/'
    vgg_path = '/home/charis/kul-thesis/SGAN/weights/vgg16_feat.pth'

    # # -----KVASIR dataset-----
    image_path = os.path.join(data_root, 'kvasir-dataset-v2/images')
    label_path = os.path.join(data_root, 'kvasir-dataset-v2/labels')
    train_file = os.path.join(data_root, 'SGAN/DSS-pytorch/train_kvasir.txt')
    # train_file = os.path.join(data_root, 'train_kvasir.txt')
    val_file = os.path.join(data_root, 'SGAN/DSS-pytorch/val_kvasir.txt')
    # val_file = os.path.join(data_root, 'val_kvasir.txt')
    
    # # # -----PASCAL VOC 2012 dataset-----
    # image_path = os.path.join(data_root, 'VOCdevkit/VOC2012/JPEGImages')
    # label_path = os.path.join(data_root, 'VOCdevkit/VOC2012/SegmentationClassAug')
    # train_file = os.path.join(data_root, 'SGAN/DSS-pytorch/train_voc.txt')
    # val_file = os.path.join(data_root, 'SGAN/DSS-pytorch/val_voc.txt')
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=256)  # 256
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--clip_gradient', type=float, default=1.0)
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--vgg', type=str, default=vgg_path)
    parser.add_argument('--train_path', type=str, default=image_path)
    parser.add_argument('--label_path', type=str, default=label_path)
    parser.add_argument('--train_file', type=str, default=train_file)
    parser.add_argument('--epoch', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=1)  # 8
    parser.add_argument('--val', type=bool, default=True)
    parser.add_argument('--val_path', type=str, default=image_path)
    parser.add_argument('--val_label', type=str, default=label_path)
    parser.add_argument('--val_file', type=str, default=val_file)
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='./results')
    parser.add_argument('--epoch_val', type=int, default=10)
    parser.add_argument('--epoch_save', type=int, default=25)
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)

    # Testing settings
    parser.add_argument('--test_path', type=str, default=image_path)
    parser.add_argument('--test_label', type=str, default=label_path)
    parser.add_argument('--test_file', type=str, default=val_file)
    parser.add_argument('--model', type=str, default='./weights/final.pth')
    parser.add_argument('--test_fold', type=str, default='./results/test')
    parser.add_argument('--use_crf', type=bool, default=False)

    # Misc
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)

    config = parser.parse_args()
    if not os.path.exists(config.save_fold):
        os.mkdir(config.save_fold)
    main(config)

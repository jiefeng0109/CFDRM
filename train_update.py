from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.resnet import resnet18
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
import pycocotools.coco as coco
from cocoeval import COCOeval
from merge import merge
import sys
from test import test
from map import voc_eval
from trains.update_annotation_1 import update_annotation  # todo
# os.environ[]


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def main(opt, dataset_name):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    # print(Dataset)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return
    # 初始化
    # train_loader = torch.utils.data.DataLoader(
    #     Dataset(opt, 'train', 0),  # todo datasets/coco.py
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    #     num_workers=opt.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )

    ap = []
    print('Starting training...')
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        # 在最后更新的，亦可在此处更新
        train_loader = torch.utils.data.DataLoader(
            Dataset(opt, 'train', epoch-1),  # todo datasets/coco.py
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True
        )
        print(epoch)
        for epoch_iter in range(1):
            # print(epoch_iter)
            log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model)

        with HiddenPrints():
            opt = opts().parse()
            opt.load_model = os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch))
            test(opt, epoch=epoch)
        print('evaluating...')
        # todo E:\Jiang\data\MOT\xx\pseudo_label\gt.json     VISO:E:\Jiang\data\RsCarData\gt.json
        if dataset_name == 'DXB' or dataset_name == 'SD' or dataset_name == 'SkySat':
            rec, prec, ap_3 = voc_eval(detpath=os.path.join(opt.save_dir, 'results_%d.json' % (epoch-1)),
                                       annopath=r'E:\Jiang\data\MOT\\'+dataset_name+r'\pseudo_label\gt.json',
                                       ovthresh=0.3)
            rec, prec, ap_5 = voc_eval(detpath=os.path.join(opt.save_dir, 'results_%d.json' % (epoch-1)),
                                       annopath=r'E:\Jiang\data\MOT\\'+dataset_name+r'\pseudo_label\gt.json',
                                       ovthresh=0.5)
        elif dataset_name == 'VISO':
            rec, prec, ap_3 = voc_eval(detpath=os.path.join(opt.save_dir, 'results_%d.json' % (epoch-1)),
                                       # annopath=r'E:\Jiang\data\MOT\\'+dataset_name+r'\pseudo_label\gt.json',
                                       annopath=r'E:\Jiang\data\RsCarData\gt.json',
                                       ovthresh=0.3)
            rec, prec, ap_5 = voc_eval(detpath=os.path.join(opt.save_dir, 'results_%d.json' % (epoch-1)),
                                       annopath=r'E:\Jiang\data\RsCarData\gt.json',
                                       # annopath=r'E:\Jiang\data\MOT\\'+dataset_name+r'\pseudo_label\gt.json',
                                       ovthresh=0.5)
        print('AP@.3: ', ap_3, 'AP@.5: ', ap_5, 'recall:', rec, 'precision:', prec)
        ap.append([ap_3, ap_5])
        np.savetxt(os.path.join(opt.save_dir, 'ap.txt'), np.array(ap), fmt='%f')

        save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # 更新标签
        with HiddenPrints():
            print('update pseudo label')
            if dataset_name == 'DXB' or dataset_name == 'SD' or dataset_name == 'SkySat':
                update_annotation(r'exp\\'+dataset_name+r'\results.json',
                                  r'E:\Jiang\data\MOT\\'+dataset_name+r'\pseudo_label\\'+'pseudo_%d.json' % (epoch-1),  # 上一个保存的
                                  r'E:\Jiang\data\MOT\\'+dataset_name+r'\pseudo_label\train',  # 这应该有两个文件
                                  r'E:\Jiang\data\MOT\\'+dataset_name+r'\pseudo_label',   # todo
                                  epoch)
            # VISO
            elif dataset_name == 'VISO':
                update_annotation(r'exp\VISO\results.json',
                                  r'E:\Jiang\data\RsCarData\\' + 'pseudo_%d.json' % (epoch - 1),  # 上一个保存的
                                  r'E:\Jiang\data\RsCarData\train',  # 这应该有两个文件
                                  r'E:\Jiang\data\RsCarData',  # todo
                                  epoch)
            # update_annotation(os.path.join(opt.save_dir, 'results.json'),
            #                   r'E:\jqp\data\MOT\DXB\pseudo.json',
            #                   r'E:\jqp\data\MOT\DXB\val',
            #                   resnet,
            #                   epoch,
            #                   device=opt.device)
            # 更新dataloader
            Dataset = get_dataset(opt.dataset, opt.task)
            opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
            # train_loader = torch.utils.data.DataLoader(
            #     Dataset(opt, 'train', epoch),
            #     batch_size=opt.batch_size,
            #     shuffle=True,
            #     num_workers=opt.num_workers,
            #     pin_memory=True,
            #     drop_last=True
            # )

    logger.close()
    np.savetxt(os.path.join(opt.save_dir, 'ap.txt'), np.array(ap), fmt='%f')


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    opt = opts().parse()
    dataset_name = 'VISO'  # todo
    main(opt, dataset_name=dataset_name)

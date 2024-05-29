"""
@Time   :   2021-01-12 15:23:56
@File   :   main.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import argparse
import os
import matplotlib as mpl
import torch, json
from transformers import BertTokenizer, BertConfig, AutoTokenizer
import pytorch_lightning as pl
from src.dataset import get_corrector_loader
from src.dfmodel import SoftMaskedBertModel as model_zl
from src.data_processor import preproc
from src.utils import get_abs_path, load_json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_device", default='cuda', type=str, help="硬件，cpu or cuda")
    parser.add_argument("--hidden_size", default=256, type=int, help="GRU隐藏层维度")
    parser.add_argument("--gpu_index", default=0, type=int, help='gpu索引, one of [0,1,2,3,...]')
    parser.add_argument("--load_checkpoint", nargs='?', const=True, default=False, type=str2bool,
                        help="是否加载训练保存的权重, one of [t,f]")
    parser.add_argument('--bert_checkpoint', default='checkpoint', type=str)
    parser.add_argument('--model_save_path', default='checkpoint', type=str)
    parser.add_argument('--epochs', default=30, type=int, help='训练轮数')
    parser.add_argument('--batch_size', default=16, type=int, help='批大小')
    parser.add_argument('--warmup_epochs', default=8, type=int, help='warmup轮数, 需小于训练轮数')
    parser.add_argument('--lr', default=2e-5, type=float, help='学习率')
    parser.add_argument('--accumulate_grad_batches',
                        default=2,
                        type=int,
                        help='梯度累加的batch数')
    parser.add_argument('--mode', default='test', type=str,
                        help='代码运行模式，以此来控制训练测试或数据预处理，one of [train, test, preproc]')
    parser.add_argument('--loss_weight', default=0.85, type=float, help='论文中的lambda，即correction loss的权重')
    arguments = parser.parse_args()
    if arguments.hard_device == 'cpu':
        arguments.device = torch.device(arguments.hard_device)
    else:
        arguments.device = torch.device(f'cuda:{arguments.gpu_index}')
    if not 0 <= arguments.loss_weight <= 1:
        raise ValueError(f"The loss weight must be in [0, 1], but get{arguments.loss_weight}")
    # print(arguments)
    return arguments


def main(args):
    # args = parse_args()
    if args.mode == 'preproc':
        print('preprocessing...')
        preproc()
        return
    torch.manual_seed(42)
    if args.hard_device == 'cuda':
        torch.cuda.manual_seed(42)
    tokenizer = BertTokenizer.from_pretrained(args.bert_checkpoint)
    model = model_zl(args, tokenizer)
    
    train_loader = get_corrector_loader(get_abs_path('data', 'train.json'),
                                        tokenizer,
                                        sort_key=lambda x: len(x["original_text"]),
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=0)
    valid_loader = get_corrector_loader(get_abs_path('data', 'dev.json'),
                                        tokenizer,
                                        sort_key=lambda x: len(x["original_text"]),
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=0)
    test_loader = get_corrector_loader(get_abs_path('data', 'test.json'),
                                       tokenizer,
                                       sort_key=lambda x: len(x["original_text"]),
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=0)
    trainer = pl.Trainer(max_epochs=args.epochs,
                         devices=None if args.hard_device == 'cpu' else [args.gpu_index],
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         fast_dev_run=False,
                         )
    model.load_from_transformers_state_dict(get_abs_path('checkpoint', 'pytorch_model.bin'))

    if args.load_checkpoint:
        model.load_state_dict(torch.load(get_abs_path('checkpoint', f'{model.__class__.__name__}_model.bin'),
                                         map_location=args.hard_device))

    if args.mode == 'train':
        trainer.fit(model, train_loader, valid_loader)

    if args.mode == 'test':
        model.load_state_dict(
            torch.load(get_abs_path('checkpoint', f'{model.__class__.__name__}_model.bin'), map_location=args.hard_device))
        trainer.test(model, test_loader)

    if args.mode == 'predict':
        model = model_zl(args, tokenizer)
        model.load_state_dict(
            torch.load(get_abs_path('checkpoint', f'{model.__class__.__name__}_model.bin'), map_location=args.hard_device))
        model.to(args.device)
        ori_text = "我努力打败数不进的风雨。"
        predict = model(ori_text)[2]
        print(predict)


if __name__ == '__main__':
    args = parse_args()
    # print(args)
    # args.mode = 'predict'
    config = BertConfig.from_pretrained(args.bert_checkpoint)
    print(config.hidden_size)
    main(args)


import os
import sys
from contextlib import contextmanager

import torch

from exp.exp_informer import ExpInformer
from utils.tools import DotDict


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def train_informer(args, supress_output=False):
    args = DotDict(args)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if supress_output:
        with suppress_stdout():
            return __train(args)

    return __train(args)


def __train(args, supress_output=False):
    best_exp = None
    val_loss_min = None

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_pl{}_dm{}_nh{}_el{}_df{}_eb{}_{}'.format(args.model,
                                                                            args.data,
                                                                            args.features,
                                                                            args.seq_len,
                                                                            args.pred_len,
                                                                            args.d_model,
                                                                            args.n_heads,
                                                                            args.e_layers,
                                                                            args.d_ff,
                                                                            args.embed,
                                                                            ii)

        exp = ExpInformer(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        model, val_loss = exp.train()
        if not val_loss_min or val_loss < val_loss_min:
            val_loss_min = val_loss
            best_exp = exp

        torch.cuda.empty_cache()

    return best_exp


if __name__ == '__main__':
    default_args = {'model': 'informer',
                    'data': 'data-fine-tuning',
                    'root_path': './data/stock',
                    'data_path': 'AAPL.csv',
                    'features': 'MS',
                    'ftr_num': 5,
                    'd_out': 1,
                    'target': 'price',
                    'freq': '15t',
                    'seq_len': 27,
                    'pred_len': 108,
                    'itr': 10,
                    'train_epochs': 10,
                    'batch_size': 6,
                    'patience': 5,
                    'learning_rate': 0.0001,
                    'loss': 'mse',
                    'lradj': 'type1',
                    'inverse': False,
                    'd_model': 512,
                    'n_heads': 10,
                    'e_layers': 6,
                    'd_ff': 2048,
                    'embed': 't2v',
                    'activation': 'gelu',
                    'padding': 0,
                    'dropout': 0.05,
                    'output_attention': False,
                    'predict': False,
                    'num_workers': 0,
                    'use_gpu': True,
                    'gpu': 0,
                    'use_multi_gpu': False,
                    'devices': '0'}

    exp = train_informer(default_args)
    print(exp.val_loss_min)
    print(exp.test())

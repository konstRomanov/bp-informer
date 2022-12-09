import os
import sys
from contextlib import contextmanager

import numpy as np
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


def run_informer(args, supress_output=False, model=None):
    args = DotDict(args)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if supress_output:
        with suppress_stdout():
            return __run(args)

    return __run(args, model)


def __run(args, model=None):
    Exp = ExpInformer
    res = []

    if model:
        return None, model, Exp(args).predict(model)

    best_model = None
    preds = None

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

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        best_model = exp.train()

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        res.append([*exp.test()])

        if args.predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            preds = exp.predict()

        torch.cuda.empty_cache()

    return np.average(np.transpose(res), axis=1), best_model, preds


if __name__ == '__main__':
    default_args = {
        'model': 'informer',

        'data': 'aapl',
        'root_path': './data/stock',
        'data_path': 'aapl.csv',
        'features': 'MS',
        'ftr_num': 8,
        'd_out': 1,
        'target': 'price',
        'freq': '15t',

        'seq_len': 48,
        'pred_len': 24,

        'itr': 5,
        'train_epochs': 8,
        'batch_size': 58,
        'patience': 5,
        'learning_rate': 0.0001,
        'loss': 'mse',
        'lradj': 'type2',
        'inverse': True,

        'd_model': 512,
        'n_heads': 16,
        'e_layers': 16,
        'd_ff': 2048,

        'embed': 't2v',
        'activation': 'relu',
        'padding': 0,
        'dropout': 0.05,

        'output_attention': False,
        'predict': False,

        'num_workers': 0,
        'use_gpu': True,
        'gpu': 0,
        'use_multi_gpu': False,
        'devices': '0'
    }

    print(run_informer(default_args)[0])

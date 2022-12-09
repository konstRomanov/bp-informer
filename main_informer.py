import argparse

import torch

from .exp.exp_informer import ExpInformer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer', help='model of experiment, options: [informer]')

parser.add_argument('--data', type=str, required=True, default='Stock', help='data')
parser.add_argument('--root_path', type=str, default='./bp-informer/data/Stock/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='Stock.csv', help='data file')
parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--ftr_num', type=int, default=7, help='number of features')
parser.add_argument('--d_out', type=int, default=7, help='dimension of informer outputs')
parser.add_argument('--target', type=str, default='price', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')

parser.add_argument('--embed', type=str, default='t2v', help='time features encoding, options:[fixed, t2v]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--predict', action='store_true', help='whether to predict unseen future data')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = ExpInformer

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
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    if args.predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()

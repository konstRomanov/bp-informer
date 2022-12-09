import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data.data_loader import DatasetAuto
from exp.exp_basic import ExpBasic
from models.model import Informer
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')


class ExpInformer(ExpBasic):
    def __init__(self, args):
        super(ExpInformer, self).__init__(args)

    def _build_model(self):
        model = Informer(
            self.args.ftr_num,
            self.args.d_out,
            self.args.pred_len,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_ff,
            self.args.dropout,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.freq
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = DatasetAuto(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            embed=args.embed,
            freq=freq,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch, batch_time) in enumerate(vali_loader):
            pred, true = self._process_one_batch(vali_data, batch, batch_time)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch, batch_time) in enumerate(train_loader):
                # print(f"TEST - Iteration data {i}; {batch_x}; {batch_y}; {batch_x_mark}; {batch_y_mark}")
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch, batch_time)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}"
                  .format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(early_stopping.best_model)
        self.best_model = early_stopping.best_model

        return self.best_model

    def test(self):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch, batch_time) in enumerate(test_loader):
            pred, true = self._process_one_batch(test_data, batch, batch_time)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape([-1, preds.shape[-2], preds.shape[-1]])
        trues = trues.reshape([-1, trues.shape[-2], trues.shape[-1]])
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}')

        return mse, mae, rmse, mape, mspe

    def predict(self, model=None):
        pred_data, pred_loader = self._get_data(flag='pred')

        if model:
            self.model.load_state_dict(model)
        else:
            self.model.load_state_dict(self.best_model)

        self.model.eval()

        preds = []

        for i, (batch, batch_time) in enumerate(pred_loader):
            pred, _ = self._process_one_batch(pred_data, batch, batch_time)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape([-1, preds.shape[-2], preds.shape[-1]])

        res = pd.DataFrame()
        res['date'] = pred_data.pred_stamp.values
        res['price'] = preds[0, :, :].flatten()
        print(res)

        return res

    def _process_one_batch(self, dataset_object, batch, batch_time):
        batch = batch.float().to(self.device)
        batch_time = batch_time.float().to(self.device)

        mask = torch.zeros if self.args.padding == 0 else torch.ones
        enc_inp = mask([batch.shape[0], self.args.pred_len, batch.shape[-1]]).float().to(self.device)
        enc_inp = torch.cat([batch[:, :self.args.seq_len, :], enc_inp], dim=1).float().to(self.device)

        # encoder
        if self.args.output_attention:
            outputs = self.model(enc_inp, batch_time)[0]
        else:
            outputs = self.model(enc_inp, batch_time)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch[:, -self.args.pred_len:, f_dim:].to(self.device)
        return outputs, batch_y

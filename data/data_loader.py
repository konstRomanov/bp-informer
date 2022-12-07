import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from torch.utils.data import Dataset

from ..utils.time_features import time_features
from ..utils.tools import StandardScaler

warnings.filterwarnings('ignore')

TRAIN_PHASE = 0
VAL_PHASE = 1
TEST_PHASE = 2
PRED_PHASE = 3

PRICES_PER_DAY = 27

class DatasetAuto(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Stock.csv',
                 target='price', scale=True, inverse=False, embed='t2v', freq='h', test=0.2):

        self.test = test

        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        # init
        assert flag in ['train', 'test', 'val', 'pred']
        phase_map = {'train': TRAIN_PHASE, 'val': VAL_PHASE, 'test': TEST_PHASE, 'pred': PRED_PHASE}
        self.model_phase = phase_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.embed = embed
        self.freq = freq

        self.scaler = StandardScaler()

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df = pd.read_csv(self.root_path + "/" + self.data_path)
        new_cols = ['date'] + list(set(df.columns.tolist()) - {self.target, 'date'}) + [self.target]
        df = df[new_cols]

        train_l = 0
        train_r = val_l = int(len(df) * (1 - self.test * 2))
        val_r = test_l = int(len(df) * (1 - self.test))
        pred_l = len(df) - self.seq_len
        test_r = pred_r = len(df)

        assert train_r >= self.seq_len and (val_r - val_l) >= self.seq_len and (test_r - test_l) >= self.seq_len

        borders_l = [train_l, val_l, test_l, pred_l]
        borders_r = [train_r, val_r, test_r, pred_r]
        border_l = borders_l[self.model_phase]
        border_r = borders_r[self.model_phase]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df.columns[1:]
            df_ftr = df[cols_data]
        else:  # self.features == 'S'
            df_ftr = df[[self.target]]

        if self.scale:
            l, r = (0, len(df_ftr)) if self.model_phase == PRED_PHASE else (borders_l[TRAIN_PHASE], borders_r[TRAIN_PHASE])
            scaler_fit_data = df_ftr[l:r]
            self.scaler.fit(scaler_fit_data.values)
            df_ftr_sc = self.scaler.transform(df_ftr.values)
        else:
            df_ftr_sc = df_ftr.values

        df_stamp = df[['date']][border_l:border_r]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.model_phase == PRED_PHASE:
            self.pred_stamp = market_range(df_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)
            tmp = df_stamp
            df_stamp = pd.DataFrame(columns=['date'])
            df_stamp.date = list(tmp.date.values) + list(self.pred_stamp)
        data_stamp = time_features(df_stamp, embed=self.embed, freq=self.freq)

        self.data_x = df_ftr_sc[border_l:border_r]
        if self.inverse:
            self.data_y = df_ftr.values[border_l:border_r]
        else:
            self.data_y = df_ftr_sc[border_l:border_r]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        seq_l = index
        seq_r = seq_l + self.seq_len
        pred_l = seq_r - self.label_len
        pred_r = seq_r + self.pred_len

        seq_x = self.data_x[seq_l:seq_r]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[pred_l:seq_r], self.data_y[seq_r:pred_r]], 0) \
                if self.model_phase != PRED_PHASE \
                else self.data_x[pred_l:seq_r]
        else:
            seq_y = self.data_y[pred_l:pred_r] if self.model_phase != PRED_PHASE else self.data_y[pred_l:seq_r]
        seq_x_stamp = self.data_stamp[seq_l:seq_r]
        seq_y_stamp = self.data_stamp[pred_l:pred_r]
        return seq_x, seq_y, seq_x_stamp, seq_y_stamp

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1 \
            if self.model_phase != PRED_PHASE \
            else len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def market_range(start, periods, freq):
    nyse = mcal.get_calendar('NYSE')
    nyse_schedule = nyse.schedule(start_date=start, end_date=start+np.timedelta64(periods//PRICES_PER_DAY+2, 'D'))
    return mcal.date_range(nyse_schedule, frequency="15min")[1:periods]

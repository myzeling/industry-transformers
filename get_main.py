import pandas as pd
import numpy as np
from datetime import datetime
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer, Informer, Autoformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import argparse
import pickle

import feather
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='every stock to predict')

    # basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Transformer',
                    help='model name, options: [ns_Transformer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='ret_next', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='M',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='fixed',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--seed', type=int, default=2023, help='random seed')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

fix_seed = args.seed
class get_main(Exp_Basic):
    def __init__(self,args,industry_stock_dict) -> None:
        super(get_main, self).__init__(args)
        self.industry_stock_dict = industry_stock_dict
        if self.args.model == 'Transformer':
            self.model_number = 3
        elif self.args.model == 'Autoformer':
            self.model_number = 0
        elif self.args.model == 'Informer':
            self.model_number = 1
    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Autoformer': Autoformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        
        return model
    def _get_data(self, flag,data_dict ,stock_id):
        data_set, data_loader = data_provider(self.args, flag, data_dict=data_dict, industry=stock_id)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def predict(self, load=True):
        for industry_keys in self.industry_stock_dict.keys():
            for stock_ids in self.industry_stock_dict[industry_keys].keys():
                stock_id = stock_ids
                pred_data, pred_loader = self._get_data(flag='pred',data_dict= self.industry_stock_dict[industry_keys] ,stock_id = stock_id)
                path = os.path.join(self.args.checkpoints,industry_keys)
                filename = os.listdir(path)
                if load:
                    best_model_path = os.path.join(path, filename[self.model_number])
                    best_model_path = os.path.join(best_model_path, 'checkpoint.pth')
                    print(best_model_path)
                    self.model.load_state_dict(torch.load(best_model_path),False)

                preds = []
                all_data = pd.DataFrame(columns=['date','id','pred'])
                self.model.eval()
                with torch.no_grad():
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float()
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                        # decoder input
                        dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        # encoder - decoder
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        pred = outputs.detach().cpu().numpy()  # .squeeze()
                        date = batch_y_mark[:, -self.args.pred_len:, :].detach().cpu().numpy()
                        def concat_date(row):
                            year, month, day = row
                            year = int(year)
                            month = int(month)
                            day = int(day)
                            date_str = f"{year}-{month:02d}-{day:02d}"
                            return date_str
                                
                        date_strings = np.apply_along_axis(concat_date, axis=2, arr=date)
                        date_strings = date_strings[..., np.newaxis]
                        tempt = pd.DataFrame(columns=['date','id','pred'])
                        tempt['date'] = date_strings.flatten()
                        tempt['id'] = stock_id
                        tempt['pred'] = pred.flatten()
                        all_data = pd.concat([all_data,tempt],axis=0)

        # result save
        folder_path = './finally_dataframe' + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        feather.write_dataframe(all_data, folder_path + 'all_data.feather')

        return
if __name__ == '__main__':
    with open('./industry_stock.pkl', 'rb') as f:
        industry_stock_dict = pickle.load(f)
    for key in list(industry_stock_dict.keys()):
        if key in ['-1','43','76','7601','81','99']:
            del industry_stock_dict[key]
    import random
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu:
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        else:
            torch.cuda.set_device(args.gpu)

    print('Args in experiment:')
    print(args)
    GET = get_main
    get = GET(args,industry_stock_dict)
    get.predict()
from models.deeprdr import DeepRDR
from models.focal_loss import FocalLossV1
from dataset.rdr_dataset import RDRDataset
from metrics.evaluate import get_metrics, get_metrics_rank
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import math
import os
import os.path as osp
import sys
from utils.index_map import get_map_index_for_sub_arr
from args import add_default_args
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

current_path = osp.dirname(osp.realpath(__file__))
save_model_data_dir = 'save_model_data'
save_data_dir = 'save_data'
log_dir = 'logs'

class Trainer(object):
    def __init__(self, **param_dict):
        self.param_dict = param_dict
        self.setup_seed(self.param_dict['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = RDRDataset(seed=self.param_dict['seed'], task=self.param_dict['task'], fold=self.param_dict['fold'])
        self.dataset.to_tensor(self.device)
        self.file_name = __file__.split('/')[-1].replace('.py', '')
        self.trainer_info = '{}_task={}_seed={}_batch={}'.format(self.file_name, self.param_dict['task'],self.param_dict['seed'], self.param_dict['batch_size'])
        self.loss_op = FocalLossV1()
        self.build_model()
        self.log = SummaryWriter(osp.join(current_path, log_dir))

    def build_model(self):
        self.model = DeepRDR(**self.param_dict).to(self.device)
        if self.param_dict['pretrain'] != '':
            self.model.load_state_dict(torch.load(self.param_dict['pretrain']), strict=False)
            print('loaded parameters successfully')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_dict['lr'])
        self.best_res = None
        self.best_epoch = 1
        self.min_dif = 1e-10
                                       
    def setup_seed(self, seed):
        seed = int(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        # torch.use_deterministic_algorithms(True)
                                       
    def iteration(self, epoch, m_list, df_data, d_graph_node_idx, m_graph_node_idx, is_training=True, shuffle=True):
        if is_training:
            self.model.train()
            df_data_p = df_data[df_data['label']==1].copy()
            df_data_n = df_data[df_data['label']==0].copy().sample(5*df_data_p.shape[0])
            df_data = pd.concat((df_data_p, df_data_n))
            bs = self.param_dict['batch_size']
        else:
            self.model.eval()
            bs = int(self.param_dict['batch_size']/4)

        m_num = m_list.shape[0]
        if shuffle is True:
            range_idx = np.random.permutation(m_num)
        else:
            range_idx = np.arange(0, m_num)

        all_pred = []

        all_label = []
        
        all_d = []
        all_m = []

        for i in range(math.ceil(m_num/bs)):
            right_bound = min((i + 1)*bs, m_num + 1)
            batch_idx = range_idx[i * bs: right_bound]
            
            b_d_idx = df_data[df_data['mesh_idx'].isin(m_list[batch_idx])]['drug_idx'].to_numpy()
            b_m_idx = df_data[df_data['mesh_idx'].isin(m_list[batch_idx])]['mesh_idx'].to_numpy()
            b_label= torch.FloatTensor(df_data[df_data['mesh_idx'].isin(m_list[batch_idx])]['label'].to_numpy()).to(self.device)
            
            d_base_num = d_graph_node_idx.shape[0]
            m_base_num = m_graph_node_idx.shape[0]
            
            d_graph_node_idx_ = np.unique(np.concatenate((d_graph_node_idx, b_d_idx), -1))
            d_graph_node_ft = self.dataset.ft_dict['d_whole'][d_graph_node_idx_]
            m_graph_node_idx_ = np.unique(np.concatenate((m_graph_node_idx, b_m_idx), -1))
            m_graph_node_ft = self.dataset.ft_dict['m_whole'][m_graph_node_idx_]

            d_graph_map_arr = get_map_index_for_sub_arr(d_graph_node_idx, np.arange(0, self.dataset.d_num))
            batch_d_node_idx_in_graph = torch.LongTensor(d_graph_map_arr[b_d_idx.astype('int64')]).to(self.device)
            m_graph_map_arr = get_map_index_for_sub_arr(m_graph_node_idx, np.arange(0, self.dataset.m_num))
            batch_m_node_idx_in_graph = torch.LongTensor(m_graph_map_arr[b_m_idx.astype('int64')]).to(self.device)


            ft_dict = {
                'd_node_ft': d_graph_node_ft,
                'm_node_ft': m_graph_node_ft,
                'd_idx': batch_d_node_idx_in_graph,
                'm_idx': batch_m_node_idx_in_graph,
                'd_idx_base': torch.LongTensor(d_graph_map_arr[d_graph_node_idx_[:d_base_num].astype('int64')]).to(self.device),
                'm_idx_base': torch.LongTensor(m_graph_map_arr[m_graph_node_idx_[:m_base_num].astype('int64')]).to(self.device),
                'd_idx_new': torch.LongTensor(d_graph_map_arr[d_graph_node_idx_[d_base_num:].astype('int64')]).to(self.device),
                'm_idx_new': torch.LongTensor(m_graph_map_arr[m_graph_node_idx_[m_base_num:].astype('int64')]).to(self.device),
                'dd': self.dataset.ft_dict['mask_dd'],
                'mm': self.dataset.ft_dict['mask_mm'],
                'dm': self.dataset.ft_dict['mask_dm'],
                'md': self.dataset.ft_dict['mask_md'],
                'd': self.dataset.num_d,
                'dn': self.dataset.num_d_n,
                'm': self.dataset.num_m,
                'mn': self.dataset.num_m_n,
                'R': self.dataset.ft_dict['R'],
            }

            pred = self.model(**ft_dict)
            pred = pred.view(-1)
            pred_ = pred.detach().to('cpu').numpy()
            pred = torch.clamp(pred, min=1.0e-10, max=(1.0 - 1e-10))
            if is_training:
                c_loss = self.loss_op(pred, b_label)
                loss = c_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # pred_ = pred.detach().to('cpu').numpy()
            b_label_ = b_label.detach().to('cpu').numpy()
            all_pred = np.hstack([all_pred, pred_])
            all_label = np.hstack([all_label, b_label_])

            all_d = np.hstack([all_d, b_d_idx])
            all_m = np.hstack([all_m, b_m_idx])
                                       
        return all_pred, all_label, all_d, all_m

    def print_res(self, res_list, epoch, task):
        if task == 'm1':
            train_aupr, valid_aupr,  \
            train_auc, valid_auc,  \
            train_f1_score, valid_f1_score,  \
            train_accuracy, valid_accuracy,  \
            train_recall, valid_recall,  \
            train_specificity, valid_specificity,  \
            train_precision, valid_precision = res_list
            
            msg_log = 'Epoch: {:03d}, ' \
                    'AUPR: Train {:.4f}, Val: {:.4f}, ' \
                    'AUC: Train {:.4f}, Val: {:.4f}, ' \
                    'F1_score: Train {:.4f}, Val: {:.4f}, ' \
                    'Accuracy: Train {:.4f}, Val: {:.4f}, ' \
                    'Recall: Train {:.4f}, Val: {:.4f},  ' \
                    'Specificity: Train {:.4f}, Val: {:.4f},  ' \
                    'Precision: Train {:.4f}, Val: {:.4f}, ' \
                .format(epoch,
                        train_aupr, valid_aupr,  \
                        train_auc, valid_auc,  \
                        train_f1_score, valid_f1_score,   \
                        train_accuracy, valid_accuracy,  \
                        train_recall, valid_recall,  \
                        train_specificity, valid_specificity,  \
                        train_precision, valid_precision,
                        )
            print(msg_log)
        elif task == 'm2':
            train_aupr, valid_aupr, test_aupr,  \
            train_auc, valid_auc, test_auc,  \
            train_f1_score, valid_f1_score, test_f1_score,  \
            train_accuracy, valid_accuracy, test_accuracy,  \
            train_recall, valid_recall, test_recall,  \
            train_specificity, valid_specificity, test_specificity,  \
            train_precision, valid_precision, test_precision = res_list

            msg_log = 'Epoch: {:03d}, ' \
                    'AUPR: Train {:.4f}, Val: {:.4f}, Test: {:.4f}, ' \
                    'AUC: Train {:.4f}, Val: {:.4f}, Test: {:.4f}, ' \
                    'F1_score: Train {:.4f}, Val: {:.4f}, Test: {:.4f}, ' \
                    'Accuracy: Train {:.4f}, Val: {:.4f}, Test: {:.4f}, ' \
                    'Recall: Train {:.4f}, Val: {:.4f}, Test: {:.4f},  ' \
                    'Specificity: Train {:.4f}, Val: {:.4f}, Test: {:.4f},  ' \
                    'Precision: Train {:.4f}, Val: {:.4f}, Test: {:.4f}, ' \
                .format(epoch,
                        train_aupr, valid_aupr, test_aupr,  \
                        train_auc, valid_auc, test_auc,  \
                        train_f1_score, valid_f1_score, test_f1_score,  \
                        train_accuracy, valid_accuracy, test_accuracy,  \
                        train_recall, valid_recall, test_recall,  \
                        train_specificity, valid_specificity, test_specificity,  \
                        train_precision, valid_precision, test_precision
                        )

        elif task == 'm3':
            valid_mrr, test_mrr, test2_mrr,  \
            valid_ndcg, test_ndcg, test2_ndcg,  \
            valid_recall, test_recall, test2_recall,  \
            valid_hitrate, test_hitrate, test2_hitrate  = res_list

            msg_log = 'Epoch: {:03d}, ' \
                    'MRR: Val: {:.4f}, Test: {:.4f}, Test2: {:.4f}, ' \
                    'NDCG: Val: {:.4f}, Test: {:.4f}, Test2: {:.4f}, ' \
                    'Recall: Val: {:.4f}, Test: {:.4f}, Test2: {:.4f}, ' \
                    'Hitrate: Val: {:.4f}, Test: {:.4f}, Test2: {:.4f}, ' \
                .format(epoch,
                        valid_mrr, test_mrr, test2_mrr,  \
                        valid_ndcg, test_ndcg, test2_ndcg,  \
                        valid_recall, test_recall, test2_recall,  \
                        valid_hitrate, test_hitrate, test2_hitrate
                        )


    def train(self, display=True):

        for epoch in range(1, self.param_dict['epoch_num'] + 1):
            # train
            train_pred, train_label, train_d, train_m = \
                self.iteration(epoch, self.dataset.train_m_list,
                               self.dataset.df_train,
                               d_graph_node_idx=self.dataset.d_base_idx,
                               m_graph_node_idx=self.dataset.m_base_idx,
                               is_training=True)
            
            if self.param_dict['task'] == 'pretrain':
                # train_aupr, train_auc, train_f1_score, train_accuracy, train_recall, train_specificity, train_precision = get_metrics(train_label, train_pred)
                train_aupr, train_auc, train_f1_score, train_accuracy, train_recall, train_specificity, train_precision = [0, 0, 0, 0, 0, 0, 0]
                
                # valid
                valid_pred, valid_label, valid_d, valid_m = \
                    self.iteration(epoch, self.dataset.valid_d,
                                self.dataset.valid_m,
                                self.dataset.valid_label,
                                d_graph_node_idx=self.dataset.d_base_idx,
                                m_graph_node_idx=self.dataset.m_base_idx,
                                is_training=False, shuffle=False)
                
                valid_aupr, valid_auc, valid_f1_score, valid_accuracy, valid_recall, valid_specificity, valid_precision = get_metrics(valid_label, valid_pred)
 
                res_list = [
                    train_aupr, valid_aupr,
                    train_auc, valid_auc,
                    train_f1_score, valid_f1_score,
                    train_accuracy, valid_accuracy,
                    train_recall, valid_recall, 
                    train_specificity, valid_specificity,
                    train_precision, valid_precision,
                ]
    
                self.log.add_scalar('valid_aupr', valid_aupr, epoch)
                self.log.add_scalar('valid_auc', valid_auc, epoch)
                self.log.add_scalar('valid_f1_score', valid_f1_score, epoch)
                self.log.add_scalar('valid_accuracy', valid_accuracy, epoch)
                self.log.add_scalar('valid_recall', valid_recall, epoch)
                self.log.add_scalar('valid_specificity', valid_specificity, epoch)
                self.log.add_scalar('valid_precision', valid_precision, epoch)
                    
                if display:
                    self.print_res(res_list, epoch, 'm1')

                if valid_auc > self.min_dif:
                    try:
                        os.remove(osp.join(current_path, save_model_data_dir, '{}_epoch={}_param.pkl'.format(self.trainer_info, self.best_epoch)))
                    except:
                        pass
                    self.min_dif = valid_auc
                    self.best_res = res_list
                    self.best_epoch = epoch
                    save_model_param_path = osp.join(current_path, save_model_data_dir, '{}_epoch={}_param.pkl'.format(self.trainer_info, self.best_epoch))
                    torch.save(self.model.state_dict(), save_model_param_path)
                    print('Best res', end='\n')
                    self.print_res(self.best_res, self.best_epoch, 'm1')

            elif (self.param_dict['task'] == 'test1') and (epoch >= 1000):
                # train_aupr, train_auc, train_f1_score, train_accuracy, train_recall, train_specificity, train_precision = get_metrics(train_label, train_pred)
                # train_aupr, train_auc, train_f1_score, train_accuracy, train_recall, train_specificity, train_precision = [0, 0, 0, 0, 0, 0, 0]

                # valid
                valid_pred, valid_label, valid_d, valid_m = \
                    self.iteration(epoch, self.dataset.valid_m_list,
                               self.dataset.df_valid,
                               d_graph_node_idx=self.dataset.d_base_idx,
                               m_graph_node_idx=self.dataset.m_base_idx,
                               is_training=False)
                
                # valid_aupr, valid_auc, valid_f1_score, valid_accuracy, valid_recall, valid_specificity, valid_precision = get_metrics(valid_label, valid_pred)

                # test
                test_pred, test_label, test_d, test_m = \
                    self.iteration(epoch, self.dataset.test_m_list,
                               self.dataset.df_test,
                               d_graph_node_idx=self.dataset.d_base_idx,
                               m_graph_node_idx=self.dataset.m_base_idx,
                               is_training=False)
                
                # test_aupr, test_auc, test_f1_score, test_accuracy, test_recall, test_specificity, test_precision = get_metrics(test_label, test_pred)
                
                # test2
                test2_pred, test2_label, test2_d, test2_m = \
                    self.iteration(epoch, self.dataset.test2_m_list,
                               self.dataset.df_test2,
                               d_graph_node_idx=self.dataset.d_base_idx,
                               m_graph_node_idx=self.dataset.m_base_idx,
                               is_training=False)
                
                # test2_aupr, test2_auc, test2_f1_score, test2_accuracy, test2_recall, test2_specificity, test2_precision = get_metrics(test_label, test_pred)
                
                # res_list = [
                #     train_aupr, valid_aupr, test_aupr, 
                #     train_auc, valid_auc, test_auc, 
                #     train_f1_score, valid_f1_score, test_f1_score, 
                #     train_accuracy, valid_accuracy, test_accuracy, 
                #     train_recall, valid_recall, test_recall, 
                #     train_specificity, valid_specificity, test_specificity, 
                #     train_precision, valid_precision, test_precision, 
                # ]
                
                
#                 self.log.add_scalar('valid_aupr', valid_aupr, epoch)
#                 self.log.add_scalar('valid_auc', valid_auc, epoch)
#                 self.log.add_scalar('valid_f1_score', valid_f1_score, epoch)
#                 self.log.add_scalar('valid_accuracy', valid_accuracy, epoch)
#                 self.log.add_scalar('valid_recall', valid_recall, epoch)
#                 self.log.add_scalar('valid_specificity', valid_specificity, epoch)
#                 self.log.add_scalar('valid_precision', valid_precision, epoch)
                
#                 self.log.add_scalar('test_aupr', test_aupr, epoch)
#                 self.log.add_scalar('test_auc', test_auc, epoch)
#                 self.log.add_scalar('test_f1_score', test_f1_score, epoch)
#                 self.log.add_scalar('test_accuracy', test_accuracy, epoch)
#                 self.log.add_scalar('test_recall', test_recall, epoch)
#                 self.log.add_scalar('test_specificity', test_specificity, epoch)
#                 self.log.add_scalar('test_precision', test_precision, epoch)
                    
#                 if display:
#                     self.print_res(res_list, epoch, 'm2')

#                 if valid_auc > self.min_dif:
#                     try:
#                         os.remove(osp.join(current_path, save_model_data_dir, '{}_epoch={}_param.pkl'.format(self.trainer_info, self.best_epoch)))
#                     except:
#                         pass
#                     self.min_dif = valid_auc
#                     self.best_res = res_list
#                     self.best_epoch = epoch
#                     save_model_param_path = osp.join(current_path, save_model_data_dir, '{}_epoch={}_param.pkl'.format(self.trainer_info, self.best_epoch))
#                     torch.save(self.model.state_dict(), save_model_param_path)
#                     # print('Best res', end='\n')
#                     # self.print_res(self.best_res, self.best_epoch, 'm2')
                    
                train_mrr, train_ndcg, train_recall, train_hitrate = [0, 0, 0, 0]
                valid_mrr, valid_ndcg, valid_recall, valid_hitrate = get_metrics_rank(valid_d, valid_m, valid_label, valid_pred, 4164)
                test_mrr, test_ndcg, test_recall, test_hitrate = get_metrics_rank(test_d, test_m, test_label, test_pred, 4164)
                test2_mrr, test2_ndcg, test2_recall, test2_hitrate = get_metrics_rank(test2_d, test2_m, test2_label, test2_pred, 4164)
                
                res_list = [
                    valid_mrr, test_mrr, test2_mrr, 
                    valid_ndcg, test_ndcg, test2_ndcg, 
                    valid_recall, test_recall, test2_recall, 
                    valid_hitrate, test_hitrate, test2_hitrate, 
                ]
                
                if display:
                    self.print_res(res_list, epoch, 'm3')
                    
            elif self.param_dict['task'] == 'test1' and epoch < 1000:
                pass

    def evaluate_model(self):
        return 0


if __name__ == '__main__':
    parser = add_default_args()
    hparams = parser.parse_args()
    param_dict = vars(hparams)
    print(param_dict)
    trainer = Trainer(**param_dict)
    trainer.train()
    trainer.log.close()
    #trainer.evaluate_model()

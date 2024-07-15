import os.path as osp
import numpy as np
import pandas as pd
import torch

current_path = osp.dirname(osp.realpath(__file__))

class RDRDataset():
    def __init__(self, seed, task, fold=1):
        self.seed = seed
        self.d_num = 4164
        self.m_num = 3759
        
        self.num_d = 4164
        self.num_d_n = 3809
        self.num_m = 2496
        self.num_m_n = 2495

        self.ft_dict = {}
        
        self.ft_dict['d_whole'] = np.load(osp.join(current_path, 'drug_feature_4164_820.npy'), allow_pickle=True)
        self.ft_dict['m_whole'] = np.load(osp.join(current_path, 'mesh_feature_3759_2495.npy'), allow_pickle=True)
        self.ft_dict['mask_dd'] = np.load(osp.join(current_path, 'mask_dd_4164_4164.npy'), allow_pickle=True)
        self.ft_dict['mask_mm'] = np.load(osp.join(current_path, 'mask_mm_2496_2496.npy'), allow_pickle=True)
        self.ft_dict['mask_dm'] = np.load(osp.join(current_path, 'mask_dm_4164_2496.npy'), allow_pickle=True)
        self.ft_dict['mask_md'] = np.load(osp.join(current_path, 'mask_md_2496_4164.npy'), allow_pickle=True)
        self.ft_dict['R'] = np.load(osp.join(current_path, 'int_4164_2496.npy'), allow_pickle=True)
        
        self.ft_dict['m_mask'] = np.load(osp.join(current_path, 'm_mask.npy'), allow_pickle=True)

        self.d_base_idx = np.load(osp.join(current_path, 'd_base_idx.npy'), allow_pickle=True)
        self.m_base_idx = np.load(osp.join(current_path, 'm_base_idx.npy'), allow_pickle=True)
        
        if task == 'pretrain':
            data_path = osp.join(current_path, 'pretrain_idx_split.csv')
            df = pd.read_csv(data_path)
            self.df_valid = df[df['split']==fold]
            self.df_train = df[df['split']!=fold]

            self.df_train_pos = self.df_train[self.df_train['label']==1]
            self.df_train_neg = self.df_train[self.df_train['label']==0]
            
            self.train_pos_d = self.df_train_pos['drug_idx'].to_numpy()
            self.train_pos_m = self.df_train_pos['mesh_idx'].to_numpy()
            self.train_pos_label = self.df_train_pos['label'].to_numpy()

            self.train_neg_d = self.df_train_neg['drug_idx'].to_numpy()
            self.train_neg_m = self.df_train_neg['mesh_idx'].to_numpy()
            self.train_neg_label = self.df_train_neg['label'].to_numpy()
            
            self.train_d = df_train['drug_idx'].to_numpy()
            self.train_m = df_train['mesh_idx'].to_numpy()
            self.train_label = df_train['label'].to_numpy()

            self.train_neg_sample_d = None
            self.train_neg_sample_m = None
            self.train_neg_sample_label = None

            self.valid_d = self.df_valid['drug_idx'].to_numpy()
            self.valid_m = self.df_valid['mesh_idx'].to_numpy()
            self.valid_label = self.df_valid['label'].to_numpy()
            
            self.train_m_list = np.unique(self.train_m)
            self.valid_m_list = np.unique(self.valid_m)

        elif task == 'test1':
            data_path = osp.join(current_path, 'rare_idx_split_0707.csv')
            df = pd.read_csv(data_path)
            self.df_test = df[df['split']==fold]
            if fold == 1:
                self.df_valid = df[df['split']==10]
                self.df_train = df[(df['split']!=1) & (df['split']!=10) & (df['split']!=11)]
            else:
                self.df_valid = df[df['split']==fold-1]
                self.df_train = df[(df['split']!=fold) & (df['split']!=(fold-1)) & (df['split']!=11)]
                
            self.df_test2 = df[df['split']==11]

            self.df_train_pos = self.df_train[self.df_train['label']==1]
            self.df_train_neg = self.df_train[self.df_train['label']==0]
            
            self.train_pos_d = self.df_train_pos['drug_idx'].to_numpy()
            self.train_pos_m = self.df_train_pos['mesh_idx'].to_numpy()
            self.train_pos_label = self.df_train_pos['label'].to_numpy()

            self.train_neg_d = self.df_train_neg['drug_idx'].to_numpy()
            self.train_neg_m = self.df_train_neg['mesh_idx'].to_numpy()
            self.train_neg_label = self.df_train_neg['label'].to_numpy()
            
            self.train_d = self.df_train['drug_idx'].to_numpy()
            self.train_m = self.df_train['mesh_idx'].to_numpy()
            self.train_label = self.df_train['label'].to_numpy()

            self.train_neg_sample_d = None
            self.train_neg_sample_m = None
            self.train_neg_sample_label = None

            self.valid_d = self.df_valid['drug_idx'].to_numpy()
            self.valid_m = self.df_valid['mesh_idx'].to_numpy()
            self.valid_label = self.df_valid['label'].to_numpy()

            self.test_d = self.df_test['drug_idx'].to_numpy()
            self.test_m = self.df_test['mesh_idx'].to_numpy()
            self.test_label = self.df_test['label'].to_numpy()
            
            self.test2_d = self.df_test2['drug_idx'].to_numpy()
            self.test2_m = self.df_test2['mesh_idx'].to_numpy()
            self.test2_label = self.df_test2['label'].to_numpy()
            
            self.train_m_list = np.unique(self.train_m) 
            self.valid_m_list = np.unique(self.valid_m)
            self.test_m_list = np.unique(self.test_m)
            self.test2_m_list = np.unique(self.test2_m)


    def to_tensor(self, device='cpu'):
        print('RDR ft to tensor', device)
        for ft_name in self.ft_dict:
            self.ft_dict[ft_name] = torch.FloatTensor(self.ft_dict[ft_name]).to(device)




import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from models.gcn_conv_input_mat import GCNConv
from models.model_utils import kernel_combination, rls_train, getGipKernel
import sys
import time

class ATTmodule(nn.Module):
    def __init__(self, channels=2, r=2):
        super(ATTmodule, self).__init__()
        inter_channels = int(channels // r)

        self.l_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        )

        self.l_att2 = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.l_att(xa)
        w = self.sigmoid(xl)
        xi = x * w + residual * (1 - w)

        xl2 = self.l_att2(xi)
        w2 = self.sigmoid(xl2)
        xo = x * w2 + residual * (1 - w2)
        return xo


class CNNmodule(nn.Module):
    def __init__(self, in_channels, out_channels, l, kernel_size=2, stride=1, padding=1):
        super(CNNmodule, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        self.out_linear = nn.Linear(l*out_channels, l)


    def forward(self, ft):
        '''
        :param protein_ft: batch*len*amino_dim
        :return:
        '''
        batch_size = ft.size()[0]

        conv_ft = self.conv(ft)
        conv_ft = self.pool(conv_ft).view(batch_size, -1)
        conv_ft = self.out_linear(conv_ft)
        return conv_ft

class DeepRDR(nn.Module):
    def __init__(self, **param_dict):
        super(DeepRDR, self).__init__()
        self.param_dict = param_dict
        self.d_dim = param_dict['drug_dim']
        self.m_dim = param_dict['mesh_dim']
        self.h_dim = param_dict['h_dim']

        self.d_linear = nn.Linear(self.d_dim, self.h_dim)
        self.m_linear = nn.Linear(self.m_dim, self.h_dim)

        self.d_adj_trans = nn.Linear(self.h_dim, self.h_dim)
        self.m_adj_trans = nn.Linear(self.h_dim, self.h_dim)
        
        self.d_gcn = GCNConv(self.h_dim, self.h_dim)
        self.m_gcn = GCNConv(self.h_dim, self.h_dim)

        self.d_cnn = CNNmodule(in_channels=1, out_channels=8, l=self.h_dim)
        self.m_cnn = CNNmodule(in_channels=1, out_channels=8, l=self.h_dim)

        self.activation = nn.ELU()
        
        for m in self.modules():
            self.weights_init(m)
            
        self.alpha = 0.5
        self.gamma = 1.0
        self.sigma = 1.0

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)


    def forward(self, **ft_dict):
        device = ft_dict['d_node_ft'].device
        
        d_node_ft = self.d_linear(ft_dict['d_node_ft'])
        d_node_ft = self.activation(d_node_ft)

        m_node_ft = self.m_linear(ft_dict['m_node_ft'])
        m_node_ft = self.activation(m_node_ft)
        
        # local d
        d_node_ft_local = self.d_cnn(d_node_ft.unsqueeze(1))
        d_node_ft_local = self.activation(d_node_ft_local)
        d_ft_local = d_node_ft_local[ft_dict['d_idx']]

        # local m
        m_node_ft_local = self.m_cnn(m_node_ft.unsqueeze(1))
        m_node_ft_local = self.activation(m_node_ft_local)
        m_ft_local = m_node_ft_local[ft_dict['m_idx']]
        
        # global d
        d_trans_ft_global = self.d_adj_trans(d_node_ft)
        d_trans_ft_global = torch.tanh(d_trans_ft_global)
        w_global = torch.norm(d_trans_ft_global, p=2, dim=-1).view(-1, 1)
        w_mat_global = w_global * w_global.t()
        d_adj_global = torch.mm(d_trans_ft_global, d_trans_ft_global.t()) / w_mat_global

        d_node_ft_global = self.d_gcn(d_node_ft, d_adj_global)
        d_node_ft_global = self.activation(d_node_ft_global)
        d_ft_global = d_node_ft_global[ft_dict['d_idx']]
        
        d_ft_all = torch.cat([d_node_ft_global, d_node_ft_local], dim=-1)

        # global m
        
        mask_dd = ft_dict['dd']
        mask_mm = ft_dict['mm']
        mask_dm = ft_dict['dm']
        mask_md = ft_dict['md']

        num_d = float(ft_dict['d'])
        num_d_n = float(ft_dict['dn'])
        num_m = float(ft_dict['m'])
        num_m_n = float(ft_dict['mn'])

        g_d = self.gamma* num_d / num_d_n
        g_m = self.gamma* num_m / num_m_n

        R = ft_dict['R']
        
        mat_d = getGipKernel(d_ft_all, False, 2 ** (-5), False)
        Kd = kernel_combination(R, mat_d, g_d, self.alpha, mask_dd)
        
        Y_single = [ft_dict['R']]   
            
        for i in ft_dict['m_idx_new']:
            m_node_ft_new = torch.cat([m_node_ft[ft_dict['m_idx_base']], m_node_ft[i].unsqueeze(-2)], -2)
            m_trans_ft_global = self.m_adj_trans(m_node_ft_new)
            m_trans_ft_global = torch.tanh(m_trans_ft_global)
            w_global = torch.norm(m_trans_ft_global, p=2, dim=-1).view(-1, 1)
            w_mat_global = w_global * w_global.t()
            m_adj_global = torch.mm(m_trans_ft_global, m_trans_ft_global.t()) / w_mat_global

            m_node_ft_global = self.m_gcn(m_node_ft_new, m_adj_global)
            m_node_ft_global = self.activation(m_node_ft_global)
            
            m_node_ft_local_new = torch.cat([m_node_ft_local[ft_dict['m_idx_base']], m_node_ft_local[i].unsqueeze(-2)], -2)
            
            m_ft_all = torch.cat([m_node_ft_global, m_node_ft_local_new], dim=-1)

            mat_m = getGipKernel(m_ft_all, False, 2 ** (-5), False)

            Km = kernel_combination(R.T, mat_m, g_m, self.alpha, mask_mm)

            Y1 = rls_train(R, mat_m, Kd, self.sigma, mask_dm, device)
            Y2 = rls_train(R.T, mat_d, Km, self.sigma, mask_md, device)

            Y = (Y1 + Y2.T) / 2.0
            Y_single.append(Y[:,-1].unsqueeze(-1))

        
        Y = torch.cat(Y_single, -1)
        pred = Y[ft_dict['d_idx'], ft_dict['m_idx']]
        pred = pred.view(-1)
        
        return pred
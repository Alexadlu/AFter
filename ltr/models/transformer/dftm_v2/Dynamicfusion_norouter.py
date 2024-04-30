import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pickle

# from .Cells import R2TCell, T2RCell, ChannelCell, SpatailCell
from .Cells_norouter import R2TCell, T2RCell, ChannelCell, SpatailCell

def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)

def unsqueeze3d(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DynamicFusion_Layer0(nn.Module):
    def __init__(self, num_cell, num_out_path, embed_size):
        super(DynamicFusion_Layer0, self).__init__()
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path
        
        self.Channel = ChannelCell(num_out_path, embed_size)
        self.Spatial = SpatailCell(num_out_path, embed_size)
        
        self.R2T_cross = R2TCell(num_out_path, embed_size)
        self.T2R_cross = T2RCell(num_out_path, embed_size)

    def forward(self, x1, x2):
        # path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0] = self.Channel(x1,x2)
        emb_lst[1] = self.Spatial(x1,x2)
        emb_lst[2] = self.R2T_cross(x1,x2)
        emb_lst[3] = self.T2R_cross(x1,x2)
        
        aggr_res_lst = []
        for i in range(self.num_out_path):
            # skip_emb = unsqueeze3d(gate_mask[:, i]) * emb_lst[0]
            res1 = 0
            res2 = 0

            for j in range(self.num_cell):

                if emb_lst[j][0].dim() == 3:
                    cur_emb_1 = emb_lst[j][0].unsqueeze(1)
                    cur_emb_2 = emb_lst[j][1].unsqueeze(1)
                else:   
                    cur_emb_1 = emb_lst[j][0]
                    cur_emb_2 = emb_lst[j][1]
                
                res1 = res1 + cur_emb_1
                res2 = res2 + cur_emb_2
                #print('res',res.shape)
            # res = res + skip_emb#.unsqueeze(1)
            aggr_res_lst.append([res1,res2])

        return aggr_res_lst
         

class DynamicFusion_Layer(nn.Module):
    def __init__(self, num_cell, num_out_path, embed_size):
        super(DynamicFusion_Layer, self).__init__()
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path

        self.Channel = ChannelCell(num_out_path, embed_size)
        self.Spatial = SpatailCell(num_out_path, embed_size)
        
        self.R2T_cross = R2TCell(num_out_path, embed_size)
        self.T2R_cross = T2RCell(num_out_path, embed_size)

    def forward(self, aggr_embed):
        emb_lst = [None] * self.num_cell

        emb_lst[0] = self.Channel(aggr_embed[0][0],aggr_embed[0][1])
        emb_lst[1] = self.Spatial(aggr_embed[1][0],aggr_embed[1][1])
        emb_lst[2] = self.R2T_cross(aggr_embed[2][0],aggr_embed[2][1])
        emb_lst[3] = self.T2R_cross(aggr_embed[3][0],aggr_embed[3][1])

        if self.num_out_path == 1:
            aggr_res_lst = []
            gate_mask_lst = []
            res1 = 0
            res2 = 0
            for j in range(self.num_cell):
                res1 += emb_lst[j][0]
                res2 += emb_lst[j][1]

            aggr_res_lst.append([res1,res2])
        else:
            #gate_mask = (sum(path_prob) < self.threshold).float()  
            aggr_res_lst = []
            for i in range(self.num_out_path):
                # skip_emb = unsqueeze3d(gate_mask[:, i]) * emb_lst[0]
                res1 = 0
                res2 = 0
                for j in range(self.num_cell):
                    res1 = res1 + emb_lst[j][0]
                    res2 = res2 + emb_lst[j][1]
                # res = res + skip_emb
                aggr_res_lst.append([res1,res2])

        return aggr_res_lst
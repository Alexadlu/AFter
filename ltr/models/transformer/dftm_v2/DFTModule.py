import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init
from .Dynamicfusion import DynamicFusion_Layer0, DynamicFusion_Layer

class DynamicFusionModule(nn.Module):
    def __init__(self, num_layer_routing=3, num_cells=4, embed_size=1024):
        super(DynamicFusionModule, self).__init__()
        self.num_cells = num_cells = 4
        self.dynamic_fusion_l0 = DynamicFusion_Layer0(num_cells, num_cells, embed_size)
        self.dynamic_fusion_l1 = DynamicFusion_Layer(num_cells, num_cells, embed_size)
        self.dynamic_fusion_l2 = DynamicFusion_Layer(num_cells, 1, embed_size)
        #total_paths = num_cells ** 2 * (num_layer_routing - 1) + num_cells
        #self.path_mapping = nn.Linear(total_paths, path_hid)
        # self.bn = nn.BatchNorm1d(opt.embed_size)

    def forward(self, x1,x2):
        #input_cat = torch.cat([input_feat_v,input_feat_i],1)
        pairs_emb_lst1 = self.dynamic_fusion_l0(x1,x2)
        pairs_emb_lst2 = self.dynamic_fusion_l1(pairs_emb_lst1)
        pairs_emb_lst3 = self.dynamic_fusion_l2(pairs_emb_lst2)
        
        feat1 = pairs_emb_lst3[0][0] #+ x1
        feat2 = pairs_emb_lst3[0][1] #+ x2
        fusion_feat = torch.cat([feat1,feat2],1)
        # lad123
        return fusion_feat
        # if self.training:
        #     return 
        
        # else:
        #     return score
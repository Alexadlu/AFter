import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from .PSA import PSA
from .Router import Router
from .SGE import SpatialGroupEnhance
from .ECAAttention import ECAAttention
from .CoTAttention import CoTAttention
from .MutualAttention import MutualAttention

# class RectifiedIdentityCell(nn.Module):
#     def __init__(self, num_out_path, embed_size):
#         super(RectifiedIdentityCell, self).__init__()
#         self.keep_mapping = nn.ReLU()
#         self.router = Router(num_out_path, embed_size)
        
#     def forward(self, x):
#         #print('x_iden',x.shape) # 15, 1024, 36, 36
#         path_prob = self.router(x)
#         emb = self.keep_mapping(x)

#         return emb, path_prob


class ChannelCell(nn.Module):
    def __init__(self, num_out_path, embed_size):
        super(ChannelCell, self).__init__()
        self.router = Router(num_out_path, embed_size*2)
        self.channel_att_1 = ECAAttention(kernel_size=3)#.cuda()
        self.channel_att_2 = ECAAttention(kernel_size=3)#.cuda()

    def forward(self, x1,x2):
        x12 = torch.cat([x1,x2],1)
        path_prob = self.router(x12)
        esa_emb1 = self.channel_att_1(x1)
        esa_emb2 = self.channel_att_2(x2)

        return [esa_emb1,esa_emb2], path_prob


class SpatailCell(nn.Module):
    def __init__(self, num_out_path, embed_size):
        super(SpatailCell, self).__init__()
        self.router = Router(num_out_path, embed_size*2)
        self.spatial_att_1 = SpatialGroupEnhance(groups=8)
        self.spatial_att_2 = SpatialGroupEnhance(groups=8)

    def forward(self, x1,x2):
        x12 = torch.cat([x1,x2],1)
        path_prob = self.router(x12)
        sge_emb1 = self.spatial_att_1(x1)
        sge_emb2 = self.spatial_att_1(x2)
        return [sge_emb1,sge_emb2], path_prob


class T2RCell(nn.Module):
    def __init__(self, num_out_path, embed_size):
        super(T2RCell, self).__init__()
        self.router = Router(num_out_path, embed_size*2)
        self.cross_att = MutualAttention(dim=embed_size)

    def forward(self, x1,x2):
        x12 = torch.cat([x1,x2],1)
        path_prob = self.router(x12)

        cross_emb = x1 + self.cross_att(x1,x2)

        return [cross_emb,x2], path_prob

class R2TCell(nn.Module):
    def __init__(self, num_out_path, embed_size):
        super(R2TCell, self).__init__()
        self.router = Router(num_out_path, embed_size*2)
        self.cross_att = MutualAttention(dim=embed_size)

    def forward(self, x1,x2):
        x12 = torch.cat([x1,x2],1)
        path_prob = self.router(x12)

        cross_emb = x2 + self.cross_att(x2,x1)

        return [x1,cross_emb], path_prob


# class IntraModelReasoningCell(nn.Module):
#     def __init__(self, opt, num_out_path):
#         super(IntraModelReasoningCell, self).__init__()
#         self.opt = opt
#         self.router = Router(num_out_path, opt.embed_size, opt.hid_router)
#         self.sa = SelfAttention(opt.embed_size, opt.hid_IMRC, opt.num_head_IMRC)

#     def forward(self, inp, stc_lens=None):
#         path_prob = self.router(inp)
#         if inp.dim() == 4:
#             n_img, n_stc, n_local, dim = inp.size()
#             x = inp.view(-1, n_local, dim)
#         else:
#             x = inp

#         sa_emb = self.sa(x)
#         if inp.dim() == 4:
#             sa_emb = sa_emb.view(n_img, n_stc, n_local, -1)
#         return sa_emb, path_prob





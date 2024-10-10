import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add, scatter_mean
import numpy as np

# from GOOD.networks.models.GINs import GINFeatExtractor
# from GOOD.networks.models.GINvirtualnode import vGINFeatExtractor

from vector_quantize_pytorch import VectorQuantize
# from .vq_update import VectorQuantize

from gnnconv import GNN_node
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class Separator(nn.Module):
    def __init__(self, args, config):
        super(Separator, self).__init__()
        if args.dataset.startswith('GOOD'):
            # GOOD
            # if config.model.model_name == 'GIN':
            #     self.r_gnn = GINFeatExtractor(config, without_readout=True)
            # else:
            #     self.r_gnn = vGINFeatExtractor(config, without_readout=True)
            emb_d = 128
        else:
            self.r_gnn = GNN_node(num_layer=args.layer, emb_dim=args.emb_dim,
                                  drop_ratio=args.dropout, gnn_type=args.gnn_type)
            emb_d = 128

        self.separator = nn.Sequential(nn.Linear(emb_d, emb_d * 2),
                                       nn.BatchNorm1d(emb_d * 2),
                                       nn.ReLU(),
                                       nn.Linear(emb_d * 2, emb_d),
                                       nn.Sigmoid())
        self.args = args

        dim = 128
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.n_output = 1
        # convolution layers
        nn1 = Sequential(Linear(128, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

    def forward(self, data):
        # if self.args.dataset.startswith('GOOD'):
        #     # DrugOOD
        #     node_feat = self.r_gnn(data=data)
        # else:
        #     # GOOD
        #     node_feat = self.r_gnn(data)
        device=('cuda:1')
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        node_feat =x
        #score[节点数,维度数]
        score = self.separator(node_feat)  # [n, d]

        # reg on score
        #pos_score_on_node [每个节点的分数]
        pos_score_on_node = score.mean(1).to(device)# [n]
        #每个分子的正负分数
        pos_score_on_batch = scatter_add(pos_score_on_node,  batch, dim=0)  # [B]
        neg_score_on_batch = scatter_add((1 - pos_score_on_node), batch, dim=0)  # [B]
        return score, pos_score_on_batch + 1e-8, neg_score_on_batch + 1e-8


class DiscreteEncoder(nn.Module):
    def __init__(self, args, config):
        super(DiscreteEncoder, self).__init__()
        self.args = args
        self.config = config
        if args.dataset.startswith('GOOD'):
            emb_dim = 128
            # if config.model.model_name == 'GIN':
            #     self.gnn = GINFeatExtractor(config, without_readout=True)
            # else:
            #     self.gnn = vGINFeatExtractor(config, without_readout=True)
            # self.classifier = nn.Sequential(*(
            #     [nn.Linear(emb_dim, config.dataset.num_classes)]
            # ))

        else:
            emb_dim = args.emb_dim
            self.gnn = GNN_node(num_layer=args.layer, emb_dim=args.emb_dim,
                                drop_ratio=args.dropout, gnn_type=args.gnn_type)
            self.classifier = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2),
                                            nn.BatchNorm1d(emb_dim * 2),
                                            nn.ReLU(),
                                            nn.Dropout(),
                                            nn.Linear(emb_dim * 2, 1))

        self.pool = global_add_pool
        self.fc1_xd = Linear(128, 128)
        self.fc2_xd = Linear(128, 128)



        self.vq = VectorQuantize(dim=128,
                                 codebook_size=12000,
                                 commitment_weight=args.commitment_weight,
                                 decay=0.9)

        self.mix_proj = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim),
                                      nn.BatchNorm1d(emb_dim),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                      nn.Linear(emb_dim, emb_dim))

        self.simsiam_proj = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2),
                                          nn.BatchNorm1d(emb_dim * 2),
                                          nn.ReLU(),
                                          nn.Linear(emb_dim * 2, emb_dim))

        dim = 128
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.n_output = 1
        # convolution layers
        nn1 = Sequential(Linear(128, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
    def vector_quantize(self, f, vq_model):
        v_f, indices, v_loss = vq_model(f)

        return v_f, v_loss

    def forward(self, data, score):
        # if self.args.dataset.startswith('GOOD'):
        #     # DrugOOD
        #     node_feat = self.gnn(data=data)
        # else:
        #     # GOOD
        #     node_feat = self.gnn(data)
        device = ('cuda:1')
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        node_feat = x

        node_v_feat, cmt_loss = self.vector_quantize(node_feat.unsqueeze(0), self.vq)
        #这行代码涉及到 squeeze 操作，它的作用通常是从张量中移除尺寸为1的维度。在这个上下文中，squeeze(0) 可能是将第一个维度（通常是 batch 维度）中为1的维度去除，使张量在该维度上更紧凑。
        #(1,27903,128) -> (27903,128)
        node_v_feat = node_v_feat.squeeze(0)
        node_res_feat = node_feat + node_v_feat
        c_node_feat = node_res_feat * score
        s_node_feat = node_res_feat * (1 - score)

        c_graph_feat =global_add_pool(c_node_feat, batch)
        c_graph_feat=F.relu(self.fc1_xd(c_graph_feat))
        c_graph_feat = F.dropout(c_graph_feat, p=0.1, training=self.training)

        s_graph_feat=global_add_pool(s_node_feat, batch)
        s_graph_feat = F.relu(self.fc2_xd(s_graph_feat))
        s_graph_feat = F.dropout(s_graph_feat, p=0.1, training=self.training)
        # c_graph_feat = self.pool(c_node_feat, data.batch)
        # s_graph_feat = self.pool(s_node_feat, data.batch)

        # c_logit = self.classifier(c_graph_feat)

        return  c_graph_feat, s_graph_feat, cmt_loss

def frobenius_dot_product(matrix_a, matrix_b):
    # Element-wise multiplication
    elementwise_product = matrix_a * matrix_b

    # Sum all elements
    result = torch.sum(elementwise_product)
    return result
class MyModel(nn.Module):
    def __init__(self, args, config):
        super(MyModel, self).__init__()
        self.args = args
        self.config = config

        self.separator = Separator(args, config)
        self.encoder = DiscreteEncoder(args, config)

        # combined layers
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 1)  # n_output = 1 for regression task

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()




    def forward(self, data):
        score, pos_score, neg_score = self.separator(data)
       #c_graph_feat(不变图级特征），s_graph_feat（虚假图级特征）  论文中是在【节点*维度】级别进行的，代码为什么是在图级别？
        c_graph_feat, s_graph_feat, cmt_loss = self.encoder(data, score)
        # reg on score  这个变成，不变特征越占总比的0.9 loss越小
        loss_reg = torch.abs(pos_score / (pos_score + neg_score) - self.args.gamma * torch.ones_like(pos_score)).mean()
        #reg on score 原文
        loss_reg_new = torch.abs(frobenius_dot_product(pos_score,torch.ones_like(pos_score)) / (data.x.shape[0]*data.x.shape[1]) - self.args.gamma )
        xc = self.fc1(c_graph_feat)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out, c_graph_feat, s_graph_feat, cmt_loss, loss_reg

    def mix_cs_proj(self, c_f: torch.Tensor, s_f: torch.Tensor):
        n = c_f.size(0)
        perm = np.random.permutation(n)
        mix_f = torch.cat([c_f, s_f[perm]], dim=-1)
        proj_mix_f = self.encoder.mix_proj(mix_f)
        return proj_mix_f


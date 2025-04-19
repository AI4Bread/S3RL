import torch
import torch.nn.functional as F
from .Optimize_prototypes import create_hypersphere
import torch.nn as nn
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import TransformerConv as TCon
import numpy as np


epsilon = 1e-16

class SingleModel(torch.nn.Module):
    def __init__(self, drop=0.0, n_head=1, hidden_dims=[], mask=0.3, replace=0.0, mask_edge=0.0, n=0, C=0):
        super().__init__()

        hidden_dims = [int(i) for i in hidden_dims]
        [in_dim, mid_dim, out_dim] = hidden_dims
        
        self.rot = Rotation_matrix(n_head*out_dim, n, C)

        self.conv1 = TCon(in_dim, mid_dim, heads=n_head)
        self.conv2 = TCon(n_head*mid_dim, out_dim, heads=n_head)
        self.conv3 = TCon(n_head*out_dim, mid_dim, heads=n_head)
        self.conv4 = TCon(n_head*mid_dim, in_dim)
        
        self.norm1 = LayerNorm(n_head*mid_dim)
        self.norm2 = LayerNorm(n_head*out_dim)
        self.norm3 = LayerNorm(n_head*mid_dim)
        self.norm4 = LayerNorm(in_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        
        self.activate = nn.ELU()
        self.drop = drop
        self.mask = mask
        self.replace = min(replace, mask)
        self.mask_token_rate = 1 - self.replace
        
        self.mask_edge = mask_edge
        
    def forward(self, x, edge_index, t):
        mask_nodes, keep_nodes = None, None
        if self.training:
            x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self.mask_token)
            edge_index = self.drop_deges(edge_index)
        x = F.dropout(x, p=self.drop, training=self.training) 

        x = F.dropout(self.norm1(self.activate(self.conv1(x, edge_index))), p=self.drop, training=self.training) 
        emb = self.norm2(self.activate(self.conv2(x, edge_index)))

        if self.training:
            emb[mask_nodes] = 0
            
        x = F.dropout(self.norm3(self.activate(self.conv3(emb, edge_index))), p=self.drop, training=self.training) 
        recon = F.dropout(self.norm4(self.activate(self.conv4(x, edge_index))), p=self.drop, training=self.training) 
        
        class_prediction = self.rot(emb, t=t)
        
        return emb, recon, keep_nodes, class_prediction


    def encoding_mask_noise(self, x, mask_token):
        num_nodes = x.shape[0]
        if self.mask>0:
            perm = torch.randperm(num_nodes, device=x.device)
            num_mask_nodes = int(self.mask * num_nodes)
            mask_nodes = perm[: num_mask_nodes]
            keep_nodes = perm[num_mask_nodes: ]

            if self.replace > 0:
                out_x = x.clone()
                perm_mask = torch.randperm(num_mask_nodes, device=x.device)
                
                num_noise_nodes = int(self.replace * num_mask_nodes)
                noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
                
                noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
                out_x[noise_nodes] = x[noise_to_be_chosen]

                token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
                out_x[token_nodes] = 0.0
                
            else:
                out_x = x.clone()
                token_nodes = mask_nodes
                out_x[mask_nodes] = 0.0

            out_x[token_nodes] += mask_token

            return out_x, (mask_nodes, keep_nodes)
        else:
            return x, ([], torch.arange(num_nodes, device=x.device))    
     
    def drop_deges(self, edge_index):
        if self.mask_edge > 0:
            num_edge = edge_index.shape[1]
            
            perm = torch.randperm(num_edge, device=edge_index.device)
            keep_edges_idx = perm[:int(num_edge*(1-self.mask_edge))]
        
            return edge_index[:, keep_edges_idx.long()]
        else:
            return edge_index
    
class Rotation_matrix(nn.Module):
    def __init__(self, dim, n, C):
        super().__init__()
        self.n = n
        
        rot = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(rot.weight)
        self.rot = nn.utils.parametrizations.orthogonal(rot)

        self.fc = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.fc.weight)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.prototypes = nn.Parameter(F.normalize(create_hypersphere(int(C), dim))).requires_grad_(False)

        
    def forward(self, emb, t):

        a_ij = self.LeakyReLU(self.fc(emb) @ self.prototypes.t())
        _, idx = torch.topk(a_ij, dim=0, k=self.n)
        p = torch.zeros_like(a_ij, device=emb.device).requires_grad_(False)
        p[idx, torch.arange(p.shape[1])] = 1/self.n

        a_ij_softmax = F.softmax(a_ij, dim=0)
        kl_loss = (p * torch.log((p+epsilon) / (a_ij_softmax+epsilon))).sum(dim=0, keepdim=True)
        
        proto_std = t * kl_loss.t() @ torch.ones(1, self.prototypes.shape[1], device=emb.device)
        eps = torch.FloatTensor(proto_std.shape).normal_().to(emb.device)
        proto_shifted = F.normalize(self.prototypes + proto_std * eps)
        
        class_prediction = F.softmax(emb @ F.normalize(self.rot(proto_shifted)).t(), dim=-1)
        return class_prediction



def Loss_recon_graph(G, G_neg, keep_nodes, h2):
    
    loss_pos = G[keep_nodes, :][:, keep_nodes] * torch.log(torch.sigmoid(h2[keep_nodes] @ h2[keep_nodes].t())+epsilon)
    loss_neg = G_neg[keep_nodes, :][:, keep_nodes] * torch.log(1 - torch.sigmoid(h2[keep_nodes] @ h2[keep_nodes].t())+epsilon)
    
    return -loss_pos-loss_neg

def train_one_epoch(fea, h4, h2, keep_nodes, class_prediction, C, N, G, G_neg, optimizer, gamma, l1, l2, scheduler):
    
    optimizer.zero_grad()
    loss_recon = ((1 - (F.normalize(h4[keep_nodes]) * fea[keep_nodes]).sum(dim=-1))**gamma).mean()
    loss_discrete = (np.sqrt(C) - torch.norm(class_prediction, p=2, dim=0).sum()/ np.sqrt(N)) / (np.sqrt(C)-1)
    loss_recon_graph = Loss_recon_graph(G, G_neg, keep_nodes, h2).mean()
    loss = loss_recon + l1 * loss_discrete + l2 * loss_recon_graph
    
    loss.backward()
    optimizer.step()  
    scheduler.step()
    return loss.detach().item()

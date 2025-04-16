import argparse
import numpy as np
from sklearn.metrics import pairwise_distances
import random
import os

epsilon = 1e-16

def parameter_setting():
    parser = argparse.ArgumentParser()
      
    # parser.add_argument('--data_path_root', type=str, default='..')
    parser.add_argument('--data_path_root', type=str, default='./Data')
    parser.add_argument('--img_path_root', default='./Img_encoder/models/', type=str)
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    
    parser.add_argument('--knn', type=int, default=5, help='Nanostring: 5, DLPFC: 7')
    parser.add_argument('--id', type=str, default='1')
    parser.add_argument('--device', type=str, default='3')
    parser.add_argument('--dataset', default='Nanostring', type=str)
    parser.add_argument('--d_emb', type=int, default=32, help='embedding dimension')
    parser.add_argument('--d_hid', type=int, default=32, help='hidden dimension')
    parser.add_argument('--drop', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--edge_img', action='store_true', help='use image edge')
    parser.add_argument('--edge_rna', action='store_true', help='use RNA edge')
    parser.add_argument('--epoch', type=int, default=2000, help='number of epochs')
    parser.add_argument('--gamma', type=float, default=1.0, help='scale for loss')
    parser.add_argument('--l1', type=float, default=0.5, help='weight for loss')
    parser.add_argument('--l2', type=float, default=10.0, help='weight for loss')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--mask', type=float, default=0.004, help='mask rate')
    parser.add_argument('--mask_edge', type=float, default=0.4, help='mask edge rate')
    parser.add_argument('--n_head', type=int, default=1, help='number of heads')
    parser.add_argument('--replace', type=float, default=0.0, help='replace rate')
    parser.add_argument('--sched', type=bool, default=True, help='use scheduler')
    parser.add_argument('--t', type=float, default=0.13, help='temperature')
    parser.add_argument('--tolerance', type=int, default=20, help='tolerance for early stopping')
    
    return parser

def seed_torch(seed=0):
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def mclust_R(array, num_cluster, modelNames='EEE', random_seed=None):
    import rpy2.robjects as robjects
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

    robjects.r('suppressMessages(library("mclust"))')
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    if random_seed is not None:
        r_random_seed = robjects.r['set.seed']
        r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(array), num_cluster, robjects.NULL, verbose=False)
    mclust_res = np.array(res[-2])
    return mclust_res

def NN_component(fea, k=1, metric='cosine', mode='and', negative_dis=False):
    if negative_dis:
        dis = -pairwise_distances(fea, metric=metric)
    else:
        dis = pairwise_distances(fea, metric=metric)
        np.fill_diagonal(dis, np.inf)
        
    idx = np.argsort(dis, axis=-1)
    affinity = np.zeros_like(dis)
    affinity[np.arange(fea.shape[0]).reshape(-1, 1), idx[:, :k]] = 1

    if mode == 'and':
        affinity = np.logical_and(affinity, affinity.T)
    if mode == 'or':
        affinity = np.logical_or(affinity, affinity.T)
        
    return affinity

def load_train_data(id='151673', knn=7, data_path='/root/GMAE/DLPFC', img_path='./', margin=25, metric='cosine', dim_RNA=3000, dataset='DLPFC', add_img_pos=True, add_rna_pos=True, return_adata=False):
    import torch
    from .load_data.load_DLPFC import load_DLPFC_data
    from .load_data.load_Nano import load_Nano_data
    from sklearn.neighbors import kneighbors_graph
    
    if dataset == 'DLPFC' or dataset == 'Human_Breast_Cancer' or dataset == 'Mouse_Brain_Anterior':
        patchs, RNA_fea, spatial_loc, gt, adata = load_DLPFC_data(id = id, path = data_path, margin=margin, dim_RNA=dim_RNA)
    elif dataset == 'Nanostring':
        RNA_fea, patchs, gt, spatial_loc = load_Nano_data(int(id), margin, data_path)
        spatial_loc = np.array(spatial_loc).astype(float).T
    else:
        raise NotImplementedError(f"{dataset} is not implemented.")
    
    RGB = np.array([i.flatten() for i in patchs])
    if os.path.exists(img_path):
        img_fea = np.load(img_path)
    else:
        img_fea = np.zeros((RGB.shape[0], 0))
        print(f"Image feature file {img_path} not found. Using zero features instead.")
        
    combined_img = np.hstack((img_fea, RGB))
    
    G_loc = kneighbors_graph(np.array(spatial_loc), n_neighbors=knn, mode='connectivity', include_self=False).toarray() ## KNN graph
    
    def build_graph(feature, knn_val, mode, neg_dis=False):
        mat = NN_component(feature, knn_val, mode=mode, metric=metric, negative_dis=neg_dis)
        np.fill_diagonal(mat, 0)
        return np.where(G_loc > 0, 0, mat)
    
    Img_near = build_graph(combined_img, knn, 'and')
    RNA_near = build_graph(RNA_fea, knn, 'and')
    Img_far  = build_graph(combined_img, 1, 'or',  neg_dis=True)
    RNA_far  = build_graph(RNA_fea, 1, 'or',  neg_dis=True)

    G_pos = G_loc.copy()
    
    if add_img_pos:
        G_pos = np.logical_or(G_loc, Img_near)
    if add_rna_pos:
        G_pos = np.logical_or(G_loc, RNA_near)
    
    G_neg = np.logical_or(Img_far, RNA_far)
    
    edge_index = np.vstack(np.nonzero(G_pos))
    edge_index = torch.from_numpy(edge_index).long()    
    
    edge_index_neg = np.vstack(np.nonzero(G_neg))
    edge_index_neg = torch.from_numpy(edge_index_neg).long()
    
    RNA_fea = torch.from_numpy(RNA_fea).float()
    img_fea = torch.from_numpy(combined_img)
    
    G = torch.from_numpy(G_pos)
    G_neg = torch.from_numpy(G_neg)
    
    if return_adata:
        return edge_index, RNA_fea, G, G_neg, gt, adata
    else:
        return edge_index, RNA_fea, G, G_neg, gt
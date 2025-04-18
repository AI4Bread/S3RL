import argparse
import numpy as np
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


    
from scipy.optimize import linear_sum_assignment


def hungarian_match(true_label, pred_label):
    l1 = list(set(true_label))
    numclass1 = len(l1)

    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('Class Not equal, Error!!!!')
        return 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
            cost[i][j] = len(mps_d)
    
    row_ind, col_ind = linear_sum_assignment(-cost)
    
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        c2 = l2[col_ind[i]]
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c
    
    return new_predict
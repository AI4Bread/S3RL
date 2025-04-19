import numpy as np
import random
import os

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
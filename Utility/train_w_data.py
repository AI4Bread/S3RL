import numpy as np
import torch
from sklearn.metrics.cluster import adjusted_rand_score as ARI
import torch.nn.functional as F
from torch import optim
from Utility.tools import seed_torch
from Utility.backbone import SingleModel, train_one_epoch
from copy import deepcopy		
from tqdm import tqdm
from Utility.tools import parameter_setting
args = parameter_setting().parse_args([])

def train_(adata, C, **cfg):
    
    seed_torch()
    
    edge_index = torch.from_numpy(np.vstack(np.nonzero(adata.obsm['G']))).long().cuda()  
    G = torch.from_numpy(adata.obsm['G']).cuda()
    G_neg = torch.from_numpy(adata.obsm['G_neg']).cuda()
    
    waiter, min_loss = 0, torch.inf
    
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()
    X = F.normalize(torch.from_numpy(adata.X).cuda(), dim=-1)
    N, C = torch.tensor(X.shape[0], dtype=torch.float).cuda(), torch.tensor(C, dtype=torch.float).cuda()
    
    assert args.d_emb % args.n_head == 0
    assert args.d_hid % args.n_head == 0
        
    model = SingleModel(drop=args.drop, 
                        n_head=args.n_head, 
                        hidden_dims=[X.shape[1], args.d_hid//args.n_head, args.d_emb//args.n_head], 
                        mask=args.mask, 
                        replace=args.replace, 
                        mask_edge=args.mask_edge, 
                        n=int(N/C), 
                        C=C).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    
    scheduler = lambda epoch :( 1 + np.cos(epoch / args.epoch * 2) ) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    
    for epoch in tqdm(range(args.epoch), desc='Training'):
        model.train()
        emb, recon, keep_nodes, class_prediction = model(X, edge_index, t=args.t)
 
        loss = train_one_epoch(args, X, recon, emb, keep_nodes, class_prediction, C, N, G, G_neg, optimizer, scheduler)
            
        if  loss < min_loss:
            min_loss = loss
            waiter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            waiter += 1
        
        if waiter >= args.tolerance:
            print('Reached the tolerance, early stop training at epoch %d' % (epoch))
            break
              
    if  waiter >= args.tolerance:  
        model.load_state_dict(best_model_state)
        
    model.eval()
    with torch.no_grad():
        emb, recon, keep_nodes, class_prediction = model(X, edge_index, t=args.t)

        pred = class_prediction.argmax(dim=-1).cpu().numpy()

    adata.obsm['X_emb'] = emb.detach().cpu().numpy()
    adata.obsm['X_recon'] = recon.detach().cpu().numpy()
    adata.obs['pred'] = pred
    return adata

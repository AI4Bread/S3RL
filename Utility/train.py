import numpy as np
import torch
from sklearn.metrics.cluster import adjusted_rand_score as ARI
import torch.nn.functional as F
from torch import optim
from Utility.utilities import seed_torch, load_train_data
from Utility.backbone import SingleModel, train_one_epoch
from copy import deepcopy		


def train_(args=None, save_ckpt=False):
    
    seed_torch()
    waiter, min_loss = 0, torch.inf
    tolerance = args.tolerance

    edge_index, fea, G, G_neg, gt = load_train_data(id=args.id, 
                                                    knn=args.knn, 
                                                    data_path=args.data_path, 
                                                    img_path=args.img_path,
                                                    margin=args.margin, 
                                                    dataset=args.dataset, 
                                                    add_img_pos=args.edge_img, 
                                                    add_rna_pos=args.edge_rna)
                                                    
    edge_index = edge_index.cuda()
    G = G.cuda()
    G_neg = G_neg.cuda()
    
    N, C = torch.tensor(gt.shape[0], dtype=torch.float).cuda(), torch.tensor(len(set(gt)), dtype=torch.float).cuda()
    
    fea = F.normalize(fea.cuda(), dim=-1)
    
    assert args.d_emb % args.n_head == 0
    assert args.d_hid % args.n_head == 0
        
    model = SingleModel(drop=args.drop, 
                        n_head=args.n_head, 
                        hidden_dims=[fea.shape[1], args.d_hid//args.n_head, args.d_emb//args.n_head], 
                        mask=args.mask, 
                        replace=args.replace, 
                        mask_edge=args.mask_edge, 
                        n=int(N/C), 
                        C=C).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    
    scheduler = lambda epoch :( 1 + np.cos(epoch / args.epoch * 2) ) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    
    for epoch in range(1, args.epoch+1):
        model.train()
        emb, recon, keep_nodes, class_prediction = model(fea, edge_index, t=args.t)
 
        loss = train_one_epoch(args, fea, recon, emb, keep_nodes, class_prediction, C, N, G, G_neg, optimizer, scheduler)
            
        if  loss < min_loss:
            min_loss = loss
            waiter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            waiter += 1
        
        if waiter >= tolerance:
            break
                    
    if  waiter >= tolerance:  
        model.load_state_dict(best_model_state)
        
    model.eval()
    with torch.no_grad():
        emb, recon, keep_nodes, class_prediction = model(fea, edge_index, t=args.t)

        pred = class_prediction.argmax(dim=-1).cpu().numpy()
        ari_pred = ARI(gt, pred)
        
    if save_ckpt:
        torch.save(model.state_dict(), args.save_path + '/model.pth')
        torch.save(emb, args.save_path + '/emb.pth')
        torch.save(recon, args.save_path + '/recon.pth')
        np.save(args.save_path + '/pred.npy', pred)
    return ari_pred, pred, emb.detach().cpu().numpy()

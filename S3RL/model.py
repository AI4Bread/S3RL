import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from .tools import seed_torch
from .backbone import SingleModel, train_one_epoch
from copy import deepcopy		
from tqdm import tqdm

class S3RL():
    def __init__(self, 
                 adata,
                 device=torch.device('cpu'),
                 d_emb=32,
                 d_hid=32,
                 drop=0.0,
                 epoch=2000,
                 gamma=1,
                 l1=0.5,
                 l2=5,
                 lr=0.0005,
                 mask=0.0,
                 mask_edge=0.0,
                 n_head=1,
                 replace=0.0,
                 t=0.1, 
                 tolerance=20,
                 n_clu=2,
                 **kwargs):
        
        """
        Args:
            adata (AnnData): The input data in AnnData format.
            device (torch.device): Device to use for training.
            d_emb (int): Dimension of the embedding.
            d_hid (int): Dimension of the hidden layer.
            drop (float): Dropout rate.
            epoch (int): Number of epochs to train.
            gamma (float): Hyperparameter for the loss function.
            l1 (float): Hyperparameter for the loss function.
            l2 (float): Hyperparameter for the loss function.
            lr (float): Learning rate.
            mask (float): Masking rate for the input data.
            mask_edge (float): Masking rate for the edges in the graph.
            n_head (int): Number of attention heads.
            replace (float): Replacement rate for the input data.
            t (float): Temperature parameter for contrastive learning.
            tolerance (int): Early stopping tolerance.
            C (int): Number of classes.
            **kwargs: Additional arguments for the model.
        """
        
        seed_torch()
        
        self.adata = adata
        self.device = device
        self.d_emb = d_emb
        self.d_hid = d_hid
        self.drop = drop
        self.epoch = epoch
        self.gamma = gamma
        self.l1 = l1
        self.l2 = l2
        self.lr = lr
        self.mask = mask
        self.mask_edge = mask_edge
        self.n_head = n_head
        self.replace = replace
        self.t = t
        self.tolerance = tolerance
        self.n_clu = n_clu
        for k, v in kwargs.items():
            setattr(self, k, v)
                 
        assert self.d_emb % self.n_head == 0
        assert self.d_hid % self.n_head == 0
        
        self.N = self.adata.X.shape[0]
        
        self.model = SingleModel(drop=self.drop,
                                 n_head=self.n_head, 
                                 hidden_dims=[self.adata.X.shape[1], self.d_hid//self.n_head, self.d_emb//self.n_head], 
                                 mask=self.mask, 
                                 replace=self.replace, 
                                 mask_edge=self.mask_edge, 
                                 n=int(self.N/self.n_clu), 
                                 C=self.n_clu).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        self.scheduler = lambda epoch :( 1 + np.cos(epoch / self.epoch * 2) ) * 0.5
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.scheduler)
        self.best_model_state = None
        
    def train(self):

        edge_index = torch.from_numpy(np.vstack(np.nonzero(self.adata.obsm['G']))).long().to(self.device)
        G = torch.from_numpy(self.adata.obsm['G']).to(self.device)
        G_neg = torch.from_numpy(self.adata.obsm['G_neg']).to(self.device)
        waiter, min_loss = 0, torch.inf
        if not isinstance(self.adata.X, np.ndarray):
            self.adata.X = self.adata.X.toarray()
        X = F.normalize(torch.from_numpy(self.adata.X).to(self.device), dim=-1)
        
        for epoch in tqdm(range(self.epoch), desc='Training the S3RL model'):
            self.model.train()
            emb, recon, keep_nodes, class_prediction = self.model(X, edge_index, t=self.t)
 
            loss = train_one_epoch(fea=X, 
                                   h4=recon, 
                                   h2=emb, 
                                   keep_nodes=keep_nodes, 
                                   class_prediction=class_prediction, 
                                   C=self.n_clu, 
                                   N=self.N, 
                                   G=G, 
                                   G_neg=G_neg, 
                                   optimizer=self.optimizer,
                                   gamma=self.gamma,
                                   l1=self.l1,
                                   l2=self.l2,
                                   scheduler=self.scheduler)
            
            if  loss < min_loss:
                min_loss = loss
                waiter = 0
                self.best_model_state = deepcopy(self.model.state_dict())
            else:
                waiter += 1
            
            if waiter >= self.tolerance:
                print('Reached the tolerance, early stop training at epoch %d' % (epoch))
                break
        
        if  waiter >= self.tolerance:
            self.model.load_state_dict(self.best_model_state)
            
        self.model.eval()
        with torch.no_grad():
            emb, recon, keep_nodes, class_prediction = self.model(X, edge_index, t=self.t)
            pred = class_prediction.argmax(dim=-1).cpu().numpy()
            
        self.adata.obsm['X_emb'] = emb.detach().cpu().numpy()
        self.adata.obsm['X_recon'] = recon.detach().cpu().numpy()
        self.adata.obs['pred'] = pred
        
        return self.adata
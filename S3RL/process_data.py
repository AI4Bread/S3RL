import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances

def process_data(adata, image=None, pixel=16, semantic_fea=None, edge_image=False, edge_gene=True, metric='cosine', knn=5):
    
    X = adata.X
    
    if semantic_fea is None:
        semantic_fea= np.zeros((X.shape[0], 0))
    
    if image is not None:
            
        try:
            patchs = [ image[int(round(px, 0))-pixel:int(round(px, 0))+pixel, int(round(py, 0))-pixel:int(round(py, 0))+pixel] for py, px in adata.obsm['spatial']]
            
        except Exception as e:
            
            patchs = []
            for py, px in adata.obsm['spatial']:
                img = image[int(round(px, 0))-pixel:int(round(px, 0))+pixel, int(round(py, 0))-pixel:int(round(py, 0))+pixel]
                if img.shape[0] < 2*pixel or img.shape[1] < 2*pixel:
                    pad_height = max(2*pixel - img.shape[0], 0)
                    pad_width = max(2*pixel - img.shape[1], 0)
                    img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
                patchs.append(img)
        combined_img = np.hstack((semantic_fea, np.array([i.flatten() for i in patchs])))
        
    else:
        combined_img = np.zeros((X.shape[0], 1))
            
    spatial_loc = adata.obsm['spatial']
    
    
    G_loc = kneighbors_graph(np.array(spatial_loc), n_neighbors=knn, mode='connectivity', include_self=False).toarray()
    
    def build_graph(feature, knn_val, mode, neg_dis=False):
        mat = NN_component(feature, knn_val, mode=mode, metric=metric, negative_dis=neg_dis)
        np.fill_diagonal(mat, 0)
        return np.where(G_loc > 0, 0, mat)

    Img_near = build_graph(combined_img, knn, 'and')
    RNA_near = build_graph(X, knn, 'and')
    Img_far = build_graph(combined_img, 1, 'or',  neg_dis=True)
    RNA_far = build_graph(X, 1, 'or',  neg_dis=True)

    G_pos = G_loc.copy()
    
    if edge_image:
        G_pos = np.logical_or(G_loc, Img_near)
    if edge_gene:
        G_pos = np.logical_or(G_loc, RNA_near)
    
    G_neg = np.logical_or(Img_far, RNA_far)
    
    adata.obsm['G'] = G_pos
    adata.obsm['G_neg'] = G_neg
    
    return adata


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
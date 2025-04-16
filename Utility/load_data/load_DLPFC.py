import numpy as np
import pandas as pd
import scanpy as sc
import cv2
import os

def load_DLPFC_data(id, path='./', dim_RNA=3000, margin=25):

    adata = sc.read_h5ad(os.path.join(path, id, 'sampledata.h5ad'))
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=dim_RNA, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    Ann_df = pd.read_csv('%s/%s/annotation.txt'%(path, id), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    adata = adata[adata.obs.notna().all(axis=1)].copy()
    sc.tl.rank_genes_groups(adata, "Ground Truth", method="wilcoxon")
    
    adata =  adata[:, adata.var['highly_variable']]

    if os.path.exists(os.path.join(path, id, 'spatial/full_image.tif')):
        image = cv2.imread(os.path.join(path,id, 'spatial/full_image.tif'))
    elif os.path.exists(os.path.join(path, id, 'spatial/tissue_hires_image.png')):
        image = cv2.imread(os.path.join(path,id, 'spatial/tissue_hires_image.png'))
        
    try:
        patchs = [ image[int(round(px, 0))-margin:int(round(px, 0))+margin, int(round(py, 0))-margin:int(round(py, 0))+margin] for py, px in adata.obsm['spatial']]
        
    except Exception as e:
        
        patchs = []
        for py, px in adata.obsm['spatial']:
            img = image[int(round(px, 0))-margin:int(round(px, 0))+margin, int(round(py, 0))-margin:int(round(py, 0))+margin]
            if img.shape[0] < 2*margin or img.shape[1] < 2*margin:
                pad_height = max(2*margin - img.shape[0], 0)
                pad_width = max(2*margin - img.shape[1], 0)
                img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
            patchs.append(img)
        

    spatial_loc = adata.obsm['spatial']
    RNA_emb = adata.X.toarray()

    labels = list(adata.obs['Ground Truth'].values)
    labels_set = set(labels)
    labels_dict = {k:v for k, v in zip(labels_set, list(range(len(labels_set))))}
    gt = np.array([labels_dict[i] for i in labels])

    return patchs, RNA_emb, spatial_loc, gt, adata



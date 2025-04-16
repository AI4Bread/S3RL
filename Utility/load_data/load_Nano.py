import pandas as pd
import numpy as np
import cv2
import os

def generate_img_feature(data, fov, margin, root_path):
    data_fov = data[data['fov'] == fov]
    data_fov = data_fov.reset_index(drop=True)

    # the x and y are reversed in the csv file
    x = data_fov['CenterY_local_px']
    y = data_fov['CenterX_local_px']
    
    big_img = cv2.imread(os.path.join(root_path, 'CellComposite/CellComposite_F{}.jpg').format(str(fov).zfill(3)))
    try:
        imgs = [big_img[x[i]-margin:x[i]+margin, y[i]-margin:y[i]+margin] for i in range(len(x))]
        
    except Exception as e:
        imgs = []
        for i in range(len(x)):
            img = big_img[x[i]-margin:x[i]+margin, y[i]-margin:y[i]+margin]
            if img.shape[0] < 2*margin or img.shape[1] < 2*margin:
                pad_height = max(2*margin - img.shape[0], 0)
                pad_width = max(2*margin - img.shape[1], 0)
                img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
            imgs.append(img)
        imgs = np.stack(imgs)
        print('///////////////////////////////////////')
    
    return imgs, [x, y]


label_map = {'tumors': 0, 'fibroblast': 1, 'lymphocyte': 2, 'Mcell': 3, 'neutrophil': 4, 'endothelial': 5, 'epithelial': 6, 'mast': 7}

def load_Nano_data(id, margin=16, root_path='./'):
    data = pd.read_csv(os.path.join(root_path, 'all_in_one.csv'))
    gene_feature_col = pd.read_csv(os.path.join(root_path, 'Lung9_Rep1_exprMat_file.csv')).columns[2:] #980 gene-related features 
    
    if data is None:
        data = pd.read_csv('all_in_one.csv')
    elif type(data) == str:
        data = pd.read_csv(data)

    patchs, spatial_loc = generate_img_feature(data, id, margin, root_path)


    y = data[data['fov'] == id]['cell_type'] # the cell type is string type
    y = y.map(label_map).to_numpy()

    gene_fea = data[data['fov'] == id][gene_feature_col].to_numpy().astype(float)

    return gene_fea, patchs, y, spatial_loc


if __name__ == '__main__':
    gene_fea, patchs, y = load_Nano_data(1, root_path='../../Data/Nanostring/')
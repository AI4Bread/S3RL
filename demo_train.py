
from Utility.Backbone import train_
from Utility.utilities import parameter_setting
import time
import os
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

margin_dict = {
    'DLPFC':16,
    'Nanostring':60,
    'Human_Breast_Cancer':10,
    'Mouse_Brain_Anterior':10,
}  

knn_dict = {
    'DLPFC':7,
    'Nanostring':5,
    'Human_Breast_Cancer':5,
    'Mouse_Brain_Anterior':5,
}  
    
if __name__ == '__main__':
    
    parser = parameter_setting()
    args = parser.parse_args()
    
    args.knn = knn_dict[args.dataset]
    
    ## set the path in your environment or you can directly use the default paths "args.img_path_root" and "args.data_path"
    args.img_path = os.path.join(args.img_path_root, args.dataset, args.id, 'img_emb.npy')
    args.data_path = os.path.join(args.data_path_root, args.dataset)
    args.margin = margin_dict[args.dataset]

    args.save_path = os.path.join('./checkpoints', args.dataset, args.id)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    ## load the best config file

    if args.id == '':
        cfg_path = os.path.join('./Best_cfg', args.dataset, args.dataset+'.yaml')
    else:
        cfg_path = os.path.join('./Best_cfg', args.dataset, args.id+'.yaml')
        
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f) 
    
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
                
    time_start = time.time()
    ari_pred, pred, emb = train_(args=args, save_ckpt=True)
    print(f'End, ID: {args.id} ARI: {ari_pred}, Time: {time.time()-time_start} s')
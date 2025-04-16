import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from torchvision import transforms
from utils import PatchDataset
from model import Model
import numpy as np
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Utility.load_data.load_Nano import load_Nano_data
from Utility.load_data.load_DLPFC import load_DLPFC_data

# train for one epoch to learn unique features
def train_one_epoch(net, data_loader, train_optimizer, temperature, batch_size):
    net.train()
    total_loss, total_num = 0.0, 1.0
    for pos_1, pos_2 in data_loader:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, data_loader):
    net.eval()
    feature_bank = []
    with torch.no_grad():
        # generate feature bank
        for data, _ in data_loader:
        # for data, _ in tqdm(data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous().detach().cpu().numpy()
      
    return feature_bank


def train_(args):
    
    dim_out = args.dim_out
    dim_mid = args.dim_mid
    temperature = round(args.temp, 1)
    batch_size = args.batch
    epochs = args.epoch
    lr = args.lr
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    if args.dataset == 'Nanostring':
        _, patchs, gt, spatial_loc = load_Nano_data(int(args.id), root_path=args.data_path, margin=args.margin)
    elif args.dataset == 'DLPFC' or args.dataset == 'Human_Breast_Cancer' or args.dataset == 'Mouse_Brain_Anterior':
        patchs, RNA_emb, spatial_loc, gt, adata = load_DLPFC_data(id=args.id, path=args.data_path, margin=args.margin)

  
    train_data = PatchDataset(patchs, train_transform)
    test_data = PatchDataset(patchs, test_transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = Model(dim_out, dim_mid).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    Path(os.path.join(args.save_path, args.dataset, args.id)).mkdir(parents=True, exist_ok=True)
    
    # for epoch in range(1, epochs + 1):
    for epoch in tqdm(range(1, epochs+1), desc='Training'):
        train_one_epoch(model, train_loader, optimizer, temperature, batch_size)
        emb = test(model, test_loader)

    torch.save(model.state_dict(), os.path.join(args.save_path, args.dataset, args.id, 'model.pth'))
    np.save(file=os.path.join(args.save_path, args.dataset, args.id, 'img_emb.npy'), arr=emb)

if __name__ == '__main__':
    
    margin_dict = {
        'DLPFC':16,
        'Nanostring':60,
        'Human_Breast_Cancer':10,
        'Mouse_Brain_Anterior':10,
    }   

    parser = argparse.ArgumentParser(description='Train SimCLR')
    
    parser.add_argument('--dataset', default='Mouse_Brain_Anterior', type=str)
    parser.add_argument('--id', default='1', type=str)
    parser.add_argument('--data_path_root', default='../Data/', type=str)
    parser.add_argument('--save_path', default='./models/', type=str, help='path to save model')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--dim_out', type=int, default=1024)
    parser.add_argument('--dim_mid', type=int, default=256)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--margin', type=int, default=10)
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.data_path = args.data_path_root + args.dataset
    args.margin = margin_dict[args.dataset]
    
    if args.dataset in ['Human_Breast_Cancer', 'Mouse_Brain_Anterior']:
        args.id = ''
        
    train_(args)

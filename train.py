import os
import random
import numpy as np
import torch
import argparse
import json
import torch.nn as nn

from mus2vec.modules.data import Mus2vecDataset
from mus2vec.modules.sampling import SamplerMus2vec
from mus2vec.mus2vec import Mus2vec
from mus2vec.modules.triplet_loss import BatchTripletLoss
import mus2vec.modules.encoder as encoder


parser = argparse.ArgumentParser(description='Mus2vec Training')
parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data_dir', default='', type=str, metavar='PATH',
                    help='absolute path to data (default: none)')
parser.add_argument('--model_dir', default='', type=str, metavar='PATH',
                    help='absolute path to model directory (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')


# Directories
root = os.path.dirname(__file__)
# model_folder = os.path.join(root,"model")
# json_dir = os.path.join(root,"data/fma_10k.json")


device = torch.device("cuda")

def train(train_loader, model, optimizer, criterion):
    loss_epoch = 0

    for idx, (x_a, x_p, x_n) in enumerate(train_loader):

        x_a = torch.reshape(x_a,(-1,x_a.shape[-2],x_a.shape[-1])).unsqueeze(1)
        x_p = torch.reshape(x_p,(-1,x_p.shape[-2],x_p.shape[-1])).unsqueeze(1)
        x_n = torch.reshape(x_n,(-1,x_n.shape[-2],x_n.shape[-1])).unsqueeze(1)

        optimizer.zero_grad()
        x_a = x_a.to(device)
        x_p = x_p.to(device)
        x_n = x_n.to(device)

        z_a, z_p, z_n = model(x_a, x_p, x_n)
        loss = criterion(z_a, z_p, z_n)

        loss.backward()

        optimizer.step()

        if idx % 10 == 0:
            print(f"Step [{idx}/{len(train_loader)}]\t Loss: {loss.item()}")
        
        

        loss_epoch += loss.item()
    
    return loss_epoch

def save_ckp(state,epoch,model_folder):
    if not os.path.exists(model_folder): 
        os.mkdir(model_folder)
    torch.save(state, "{}/model_ver1_epoch_{}.pth".format(model_folder,epoch))

def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['loss']

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def load_index(dirpath):
    dataset = {}
    idx = 0
    print('=>Loading indices of train data')
    json_path = os.path.join(root, dirpath.split('/')[-1] + ".json")

    for filename in os.listdir(dirpath):
      if filename.endswith(".npy"): 
        dataset[idx] = filename
        idx += 1
    with open(json_path, 'w') as fp:
        json.dump(dataset, fp)

    return json_path


def main():
    args = parser.parse_args()
    data_dir = args.data_dir
    model_folder = args.model_dir
    json_dir = load_index(data_dir)
    
    # Hyperparameters
    batch_size = 6
    n_triplets = 16
    learning_rate = 1e-4
    num_epochs = args.epochs

    if args.seed is not None:
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_dataset = Mus2vecDataset(path=data_dir, json_dir=json_dir, sampler=SamplerMus2vec(n_triplets=n_triplets, train=True, bias=True))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)

    model = Mus2vec(encoder=encoder.Encoder())
    model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 240, eta_min = 1e-7)
    criterion = BatchTripletLoss(batch_size=batch_size, n_triplets=n_triplets, margin = 0.1)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model, optimizer, scheduler, start_epoch, loss_log = load_ckp(args.resume, model, optimizer, scheduler)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    else:
        start_epoch = 0
        loss_log = []
        
    

    
    best_loss = train(train_loader, model, optimizer, criterion)
    
    # training
    model.train()
    for epoch in range(start_epoch+1, num_epochs+1):
        print("#######Epoch {}#######".format(epoch))
        loss_epoch = train(train_loader, model, optimizer, criterion)
        loss_log.append(loss_epoch)
        if loss_epoch < best_loss and epoch%10==0:
            best_loss = loss_epoch
            
            checkpoint = {
                'epoch': epoch,
                'loss': loss_log,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            save_ckp(checkpoint,epoch,model_folder)
        scheduler.step()
    
  
        
if __name__ == '__main__':
    main()
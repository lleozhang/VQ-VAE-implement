from model.VQ_VAE import VQ_VAE
from data.data import ImageSet
import random
import numpy as np
import torch
import os, shutil
from copy import deepcopy
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
from torchvision.transforms import Normalize
import torch.distributed as dist
from torchvision.utils import save_image

#dist.init_process_group(backend='nccl')

def set_seed(seed = 1008):
    seed = np.random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def build_dir(log_path, save_path):
    try:
        os.mkdir(log_path)
    except Exception as e:
        pass

    try:
        os.mkdir(save_path)
    except Exception as e:
        pass

def train_model(model_path, dataset, batch_size, num_epochs, 
                        lr, device, log_file, local_rank):
    
    model = torch.load(model_path, map_location = 'cpu')
    model.train()
    Loss = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr = lr)
    best_train = 1e10
    trans = Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    train_sampler = None
    if device != 'cpu':
        torch.cuda.set_device(local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)   
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids = [local_rank], find_unused_parameters=True)

    for epoch in trange(num_epochs):
        train_sampler.set_epoch(epoch)
        log_file.write(str(epoch)+':\n')

        total_train_loss = 0

        if train_sampler:
            dataloader = DataLoader(dataset, batch_size = batch_size, 
                                drop_last = True, num_workers= 2, sampler = train_sampler)
        else:
            dataloader = DataLoader(dataset, batch_size = batch_size, 
                                shuffle = True, drop_last = True, num_workers= 2)
        
        
        for input_img in tqdm(dataloader):
            if device != 'cpu':
                input_img = input_img.cuda(non_blocking = True)
            output, loss = model(trans(input_img))
            batch_loss = Loss(output, input_img) + loss.mean(dim = 0)
            total_train_loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        
        if total_train_loss < best_train:
            best_train = total_train_loss
            torch.save(model.module, model_path)
        if local_rank == 0:
            print(total_train_loss)
            log_file.write('loss = ' + str(total_train_loss) + '\n')
        #print(total_train_loss)
    

def evaluate(model_path, dataset, device):
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size = 1, 
                                shuffle = True, drop_last = True)
        trans = Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        for input_img in dataloader:
            input_img = input_img.to(device)
            output, loss = model(trans(input_img))
            save_image(output.view(3, 224, 224).data, "gen.png" , nrow=1, normalize = True)
            save_image(input_img.view(3, 224, 224).data, "ori.png", nrow = 1, normalize = True)
            break 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train')
    parser.add_argument('--batch_size', default = 32, type = int)
    parser.add_argument('--seed', default = 1008, type = int)
    parser.add_argument('--epochs', default = 100, type = int)
    parser.add_argument('--lr', default = 0.0001, type = float)
    parser.add_argument('--dropout', default = 0.5, type = float)
    parser.add_argument('--token_dim', default = 1280, type = int)
    parser.add_argument('--token_siz', default = 8192, type = int)
    parser.add_argument('--num_heads', default = 4, type = int)
    parser.add_argument('--medium_dim', default = 128, type = int)
    parser.add_argument('--device', default = 'cuda', type = str)
    parser.add_argument('--log_path', default = 'train_logs/', type = str)
    parser.add_argument('--save_path', default = 'model_file/', type = str)
    parser.add_argument('--resume', action = 'store_true')
    parser.add_argument('--type', default = 'train', type = str)
    parser.add_argument('--test_path', default = 'test.txt', type = str)
    parser.add_argument('--name', default = 'VQ_VAE', type = str)
    parser.add_argument('--local_rank', default = -1, type = int)
    args = parser.parse_args()
    
    set_seed(args.seed)
    build_dir(args.log_path, args.save_path)

    device = args.device
    
    model_path = args.save_path + args.name + '.pkl'
    
    if args.type == "test":
        dataset = ImageSet('example_data/')
        evaluate(model_path, dataset, args.device)
    else:
        dataset = ImageSet('example_data/')
        if not args.resume:
            model = VQ_VAE(args.token_siz, args.token_dim, args.medium_dim, args.dropout, args.num_heads)
            torch.save(model, model_path)
            log = open(args.log_path + args.name + '_train_logs. txt', 'w')
        else:
            log = open(args.log_path + args.name + '_train_logs. txt', 'w+')
            log.write('resuming...' + '\n')
            
        train_model(model_path, dataset, batch_size = args.batch_size, 
                    num_epochs = args.epochs, lr = args.lr, device = device, 
                    log_file = log, local_rank = args.local_rank)
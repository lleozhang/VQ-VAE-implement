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

def evaluate(model, dataset, device, epoch, log):
    model.eval()
    Loss = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size = 1, 
                                shuffle = True, drop_last = True)
        #trans = Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        flag = 1
        for input_img in tqdm(dataloader):
            input_img = input_img.to(device)
            output, loss = model(input_img, 0)
            batch_loss = Loss(output, input_img) + loss
            total_loss += batch_loss.item()
            if flag:
                save_image(output.view(3, 224, 224).data, "saved_imgs/gen%d.png" % epoch , nrow=1, normalize = True)
                save_image(input_img.view(3, 224, 224).data, "saved_imgs/ori%d.png" % epoch, nrow = 1, normalize = True)
            flag = 0 
    if log:
        log.write('val loss: %f' % total_loss)
    return total_loss


def train_model(model, train_set, val_set, batch_size, num_epochs, 
                    lr, device, log_path, local_rank, with_sync,
                    resume_path, save_path):
    
    start_epoch = 0
    best_val = 1e10
    best_train = 1e10
    Loss = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), 
                                 lr = lr)
    if resume_path:
        checkpoint = torch.load(resume_path, map_location = 'cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        log_file = open(checkpoint['log_file'], 'w+')
        best_val = checkpoint['val_loss']
    else:
        log_file = open(log_path, 'w')
        resume_path = save_path + 'best.pth'
    
    #trans = Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    train_sampler = None
    if with_sync:
        torch.cuda.set_device(local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)   
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids = [local_rank], find_unused_parameters=True)
    
    
    for epoch in trange(start_epoch, start_epoch + num_epochs):
        model.train()
        model = model.to(device)
        log_file.write(str(epoch)+':\n')

        total_train_loss = 0

        if train_sampler:
            train_sampler.set_epoch(epoch)
            dataloader = DataLoader(train_set, batch_size = batch_size, 
                                drop_last = True, num_workers= 2, sampler = train_sampler)
        else:
            dataloader = DataLoader(train_set, batch_size = batch_size, 
                                shuffle = True, drop_last = True, num_workers= 2)
        
        debug = 0
        for input_img in tqdm(dataloader):
            if with_sync:
                input_img = input_img.cuda(non_blocking = True)
            else:
                model = model.to(device)
                input_img = input_img.to(device)
            output, loss = model(input_img, debug)
            batch_loss = Loss(output, input_img) + loss
            total_train_loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            debug = 0
        
        if with_sync:
            if local_rank == 0:
                val_loss = evaluate(model, val_set, device, epoch)
        else:
            val_loss = evaluate(model, val_set, device, epoch, log_file)
        
        if with_sync:
            state = {
                        'epoch' : epoch,
                        'train_loss' : total_train_loss,
                        'val_loss' : val_loss,
                        'state_dict' : model.module.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'log_file' : log_path,
                    }
        else:
            state = {
                        'epoch' : epoch,
                        'train_loss' : total_train_loss,
                        'val_loss' : val_loss,
                        'state_dict' : model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'log_file' : log_path,
                    }
        if with_sync:
            if local_rank == 0:
                torch.save(state, save_path + str(epoch) + '.pth')
        else:
            torch.save(state, save_path + str(epoch) + '.pth')
        
        if val_loss < best_val:
            best_val = val_loss
            if with_sync:
                if local_rank == 0:
                    torch.save(state, resume_path)
            else:
                torch.save(state, resume_path)
    
        if with_sync:
            if local_rank == 0:
                log_file.write('train_loss = ' + str(total_train_loss) + '\n')
        else:
            log_file.write('train_loss = ' + str(total_train_loss) + '\n')
    

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
    parser.add_argument('--type', default = 'train', type = str)
    parser.add_argument('--test_path', default = 'test.txt', type = str)
    parser.add_argument('--name', default = 'VQ_VAE', type = str)
    parser.add_argument('--local_rank', default = -1, type = int)
    parser.add_argument('--with_sync', action = 'store_true')
    parser.add_argument('--resume_type', default = None, type = str)
    args = parser.parse_args()
    
    if args.with_sync:
        dist.init_process_group(backend='nccl')
    
    set_seed(args.seed)

    device = args.device
    default_path = '/data/zhanghuixuan/multimodal_gen/model_file/'
    
    save_path = default_path + args.name
    
    if args.resume_type:
        resume_path = save_path + args.resume_type + '.pth'
    elif args.type == 'test':
        resume_path = save_path + 'best' + '.pth'
    else:
        resume_path = None

    model = VQ_VAE(args.token_siz, args.token_dim, args.medium_dim, 
                           args.dropout, args.num_heads)

    if args.type == "test":
        dataset = ImageSet('/data/zhanghuixuan/multimodal_gen/test_imgs/')
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        evaluate(model, dataset, args.device, 0, None)
    else:
        train_set = ImageSet('/data/zhanghuixuan/multimodal_gen/train_imgs/')
        val_set = ImageSet('/data/zhanghuixuan/multimodal_gen/valid_imgs/')
        train_model(model, train_set, val_set, batch_size = args.batch_size, 
                    num_epochs = args.epochs, lr = args.lr, device = device, 
                    log_path = args.log_path + args.name + '_train_logs.txt', 
                    local_rank = args.local_rank, with_sync = args.with_sync, 
                    resume_path =  resume_path, save_path = save_path)